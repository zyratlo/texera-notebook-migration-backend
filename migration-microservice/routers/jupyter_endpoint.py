from uuid import uuid4
import os
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from . import resources

router = APIRouter(prefix="/jupyter", tags=["jupyter"])


@router.post("/set_notebook")
async def set_notebook(request: Request):
    try:
        data = await request.json()
        notebook_name = data.get('notebookName', 'notebook.ipynb')
        notebook_data = data.get('notebookData')

        if not notebook_data:
            raise HTTPException(status_code=400, detail="Notebook data is required")

        for cell in notebook_data["cells"]:
            if 'metadata' not in cell:
                cell['metadata'] = {}
            cell['metadata']['uuid'] = str(uuid4())

        # Save the notebook JSON file
        notebook_file_path = os.path.join(resources.NOTEBOOK_PATH, notebook_name)
        with open(notebook_file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=4)

        resources.NOTEBOOK_NAME = notebook_name
        resources.NOTEBOOK_SAVED = True

        return JSONResponse(content={
            "message": "Notebook saved successfully",
            "notebookPath": notebook_file_path
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
