from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os
import json
import nbformat as nbf
import openai
from uuid import uuid4
from time import sleep, time
import asyncio
from . import resources

router = APIRouter(prefix="/openai", tags=["openai"])

openai.api_key = os.getenv("OPENAI_API_KEY")
assistant = openai.beta.assistants.create(
    name="Notebook to Texera Migrator",
    instructions=f"{resources.texera_overview}\n"
                 f"{resources.tuple_documentation}\n"
                 f"{resources.table_documentation}\n"
                 f"{resources.operator_documentation}\n"
                 f"{resources.example_of_good_conversion}\n"
                 f"{resources.visualizer_documentation}\n"
                 f"{resources.udf_input_port_documentation}\n"
                 f"{resources.example_of_multiple_udf_conversion}",
    model="o3-mini-2025-01-31"

)

assistant_id = assistant.id
thread = openai.beta.threads.create()
thread_id = thread.id


def call_gpt_assistant(prompt: str):
    openai.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )
    run = openai.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id
    )

    # Wait for completion
    while True:
        run_status = openai.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run_status.status == "completed":
            break
        elif run_status.status == "failed":
            raise Exception("Run failed!")
        asyncio.sleep(1)

    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    # Get the most recent assistant message
    for message in messages.data:
        if message.role == "assistant":
            return message.content[0].text.value


def python_script_to_texera(python_script: str, do_print: bool) -> (str, str):
    final_prompt = f"{resources.workflow_prompt}\n{python_script}"

    # Get workflow
    if do_print:
        print("Getting workflow from OpenAI Assistant API...", end="")
    response = call_gpt_assistant(final_prompt)
    if do_print:
        print("DONE")

    # Get mapping
    if do_print:
        print("Getting mapping from OpenAI Assistant API...", end="")
    mapping = call_gpt_assistant(resources.mapping_prompt)
    if do_print:
        print("DONE")

    return response, mapping


def parse_workflow_and_mapping(open_ai_workflow, open_ai_mapping):
    """
    Converts the OpenAI workflow JSON into a Texera-compatible JSON and uni-directional mapping to bidirectional mapping
    :param open_ai_workflow: OpenAI workflow
    :param open_ai_mapping: uni-directional mapping
    :return: Texera JSON and bidirectional mapping
    """
    udf_open_ai_response = json.loads(open_ai_workflow.strip("```json").strip("```").strip(), strict=False)

    workflow_json = {
        "operators": [],
        "operatorPositions": {},
        "links": [],
        "commentBoxes": [],
        "settings": {
            "dataTransferBatchSize": 400
        }
    }

    udf_mapping_to_uuid = {}

    for i, (UDF_ID, UDF_code) in enumerate(udf_open_ai_response["code"].items(), start=1):
        udf_uuid = f"PythonUDFV2-operator-{str(uuid4())}"
        udf_mapping_to_uuid[UDF_ID] = udf_uuid
        try:
            udf_output_columns = [{"attributeName": attr, "attributeType": "binary"} for attr in
                                  udf_open_ai_response["outputs"][UDF_ID]]
        except KeyError:
            udf_output_columns = []

        # Add UDF to operators
        workflow_json["operators"].append(
            {
                "operatorID": f"{udf_uuid}",
                "operatorType": "PythonUDFV2",
                "operatorVersion": "3d69fdcedbb409b47162c4b55406c77e54abe416",
                "operatorProperties": {
                    "code": UDF_code,
                    "workers": 1,
                    "retainInputColumns": False,
                    "outputColumns": udf_output_columns
                },
                "inputPorts": [
                    {
                        "portID": "input-0",
                        "displayName": "",
                        "allowMultiInputs": True,
                        "isDynamicPort": False,
                        "dependencies": []
                    }
                ],
                "outputPorts": [
                    {
                        "portID": "output-0",
                        "displayName": "",
                        "allowMultiInputs": False,
                        "isDynamicPort": False
                    }
                ],
                "showAdvanced": False,
                "isDisabled": False,
                "customDisplayName": UDF_ID,
                "dynamicInputPorts": True,
                "dynamicOutputPorts": True
            }
        )

        # Add UDF to operatorPositions
        workflow_json["operatorPositions"][udf_uuid] = {"x": 140 * i, "y": 0}

    # Add links/edges
    for source, target in udf_open_ai_response["edges"]:
        workflow_json["links"].append(
            {
                "linkID": f"link-{str(uuid4())}",
                "source": {
                    "operatorID": udf_mapping_to_uuid[source],
                    "portID": "output-0"
                },
                "target": {
                    "operatorID": udf_mapping_to_uuid[target],
                    "portID": "input-0"
                }
            }
        )

    # Parses the mapping
    mapping = json.loads(open_ai_mapping.strip("```json\n").strip("```"))

    udf_to_cell = {}
    cell_to_udf = {}
    for udf, cells in mapping.items():
        udf_uuid = udf_mapping_to_uuid[udf]
        udf_to_cell[udf_uuid] = cells
        for cell in cells:
            if cell not in cell_to_udf:
                cell_to_udf[cell] = [udf_uuid]
            else:
                cell_to_udf[cell].append(udf_uuid)

    combined_mapping = {
        "operator_to_cell": udf_to_cell,
        "cell_to_operator": cell_to_udf
    }

    return workflow_json, combined_mapping


@router.post("/get_openai_response")
def get_openai_response(do_print=True):
    """
    Calls OpenAI with Texera documentation to generate and parse a workflow JSON and mapping JSON
    :param do_print: flag that controls whether the program prints which step it is on
    :return: a JSON representing the workflow, and a JSON representing the mapping
    """

    try:
        # Load the notebook
        while not resources.NOTEBOOK_SAVED:
            sleep(2)
        notebook_path = os.path.join(resources.NOTEBOOK_PATH, resources.NOTEBOOK_NAME)
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_file = nbf.read(f, as_version=4)

        # Extract code cells and maintain separation
        code_cells = [cell for cell in notebook_file.cells if cell.cell_type == 'code']

        # Join the code into one string
        notebook_string = "\n\n".join(
            f"# START {cell['metadata']['uuid']}\n"
            f"{cell['source']}\n"
            f"# END {cell['metadata']['uuid']}"
            for cell in code_cells
        )

        start_time = time()

        open_ai_workflow, open_ai_mapping = python_script_to_texera(notebook_string, do_print)

        end_time = time()
        if do_print:
            print(f"Time taken: {end_time - start_time:.6f} seconds")

        final_workflow, final_mapping = parse_workflow_and_mapping(open_ai_workflow, open_ai_mapping)

        if do_print:
            print(final_workflow)
            print("-------------")
            print(final_mapping)

        return JSONResponse(content={"workflow": final_workflow, "mapping": final_mapping})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
