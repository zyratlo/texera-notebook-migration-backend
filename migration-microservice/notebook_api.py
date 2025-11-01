from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import os
import json
import nbformat as nbf
import openai
from uuid import uuid4
from time import sleep, time

app = Flask(__name__)
CORS(app)  # Allow all origins

NOTEBOOK_PATH = "/home/jovyan/work"
NOTEBOOK_NAME = ""
NOTEBOOK_SAVED = False

# TEXERA DOCUMENTATION

# https://github.com/Texera/texera/wiki/Guide-to-Use-a-Python-UDF
texera_overview = """
You are a robust compiler that takes python code and translates it to our personal workflow environment Texera that uses python. 

Texera is a data analytics tool that uses workflows to do machine learning and data analytics computation. User's are able to drag and drop operators and connect their inputs and outputs in a workflow graphical user interface, which the code we are going to create.

    Texera is able to use Python user defined functions. Documentation of a Python UDF in Texera follows:
    Process Data APIs

    There are three APIs to process the data in different units.

        Tuple API.

    class ProcessTupleOperator(UDFOperatorV2):

        def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
            yield tuple_

    Tuple API takes one input tuple from a port at a time. It returns an iterator of optional TupleLike instances. A TupleLike is any data structure that supports key-value pairs, such as pytexera.Tuple, dict, defaultdict, NamedTuple, etc.

    Tuple API is useful for implementing functional operations which are applied to tuples one by one, such as map, reduce, and filter.

        Table API.

    class ProcessTableOperator(UDFTableOperator):

        def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
            yield table

    Table API consumes a Table at a time, which consists of all the tuples from a port. It returns an iterator of optional TableLike instances. A TableLike is a collection of TupleLike, and currently, we support pytexera.Table and pandas.DataFrame as a TableLike instance. More flexible types will be supported down the road.

    Table API is useful for implementing blocking operations that will consume all the data from one port, such as join, sort, and machine learning training.

        Batch API.

    class ProcessBatchOperator(UDFBatchOperator):

        BATCH_SIZE = 10

        def process_batch(self, batch: Batch, port: int) -> Iterator[Optional[BatchLike]]:
            yield batch

    Batch API consumes a batch of tuples at a time. Similar to Table, a Batch is also a collection of Tuples; however, its size is defined by the BATCH_SIZE, and one port can have multiple batches. It returns an iterator of optional BatchLike instances. A BatchLike is a collection of TupleLike, and currently, we support pytexera.Batch and pandas.DataFrame as a BatchLike instance. More flexible types will be supported down the road.

    The Batch API serves as a hybrid API combining the features of both the Tuple and Table APIs. It is particularly valuable for striking a balance between time and space considerations, offering a trade-off that optimizes efficiency.

    All three APIs can return an empty iterator by yield None.

    The template code for a Python UDF follows: MAKE SURE TO USE THE CLASS NAMES AND FUNCTIONS DEFINED, THIS IS A MUST FOR THE PROGRAM TO WORK. SELECT 1 OUT OF THE 3 PROCESSING OPERATOR FUNCTIONS TO BUILD DEPENDINGO ON THE CONTEXT OF CODE TRANSLATION. 
    # Choose from the following templates:
    # 
    # from pytexera import *
    # 
    # class ProcessTupleOperator(UDFOperatorV2):
    #     
    #     @overrides
    #     def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
    #         yield tuple_
    # 
    # class ProcessBatchOperator(UDFBatchOperator):
    #     BATCH_SIZE = 10 # must be a positive integer
    # 
    #     @overrides
    #     def process_batch(self, batch: Batch, port: int) -> Iterator[Optional[BatchLike]]:
    #         yield batch
    # 
    # class ProcessTableOperator(UDFTableOperator):
    # 
    #     @overrides
    #     def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:
    #         yield table
"""

# https://github.com/Texera/texera/blob/1fa249a9d55d4dcad36d93e093c2faed5c4434f0/core/amber/src/main/python/core/models/tuple.py
tuple_documentation = """### **<code>Tuple</code> Class Overview**

The `Tuple` class is a **lazy-evaluated** data structure designed for efficient field storage and access. It provides:

1. **Support for Multiple Data Sources**:
    * Can be initialized from a `TupleLike` object, such as a `pandas.Series`, `OrderedDict`, or another `Tuple` instance.
    * Works with `ArrowTableTupleProvider` to access `pyarrow.Table` data.
2. **Lazy Field Evaluation**:
    * Field values can be either **directly stored values** or **lazy accessors** (`field_accessor`).
    * If a field is accessed and is an accessor, it is evaluated and cached.
3. **Schema (<code>Schema</code>) Enforcement**:
    * A `Tuple` can be created without a schema but can be **finalized** with one using `finalize(schema)`, which:
        * **Casts field values** (e.g., `NaN → None`, `Object → Bytes`).
        * **Validates field completeness**, ensuring all fields match the `Schema`.
4. **Pythonic Access Patterns**:
    * **Index-based access**: `tuple["field_name"]` or `tuple[index]` retrieves field values.
    * **Dictionary-like operations**: `tuple.as_dict()` returns an `OrderedDict`, and `tuple.as_series()` converts to a `pandas.Series`.
    * **Iterable support**: `for field in tuple` iterates over field values.
5. **Hashing and Comparisons**:
    * Implements `__hash__` using a Java-like hashing algorithm, allowing usage as dictionary keys.
    * Implements `__eq__`, supporting equality checks based on field contents.
6. **Partial Data Extraction**:
    * `tuple.get_partial_tuple(attribute_names)` returns a new `Tuple` instance containing only the specified fields.
"""

# https://github.com/Texera/texera/blob/1fa249a9d55d4dcad36d93e093c2faed5c4434f0/core/amber/src/main/python/core/models/table.py
table_documentation = """### **<code>Table</code> Class Overview**

The `Table` class extends `pandas.DataFrame`, providing **structured Tuple-based data management**. It is designed to integrate seamlessly with `Tuple` objects.

#### **Key Features:**

1. **Flexible Construction:**
    * Can be initialized from various sources:
        * Another `Table` (`from_table(table)`)
        * A `pandas.DataFrame` (`from_data_frame(df)`)
        * A list/iterator of `TupleLike` objects (`from_tuple_likes(tuple_likes)`)
    * Ensures all `Tuple` objects in a `Table` have **consistent field names**.
2. **Tuple Conversion:**
    * `as_tuples()`: Converts the table rows into an **iterator of <code>Tuple</code> instances**, preserving the row order.
3. **Equality Comparison (<code>__eq__</code>):**
    * Supports **row-wise equality checks** by comparing the underlying `Tuple` objects.
4. **Universal Tuple Output (<code>all_output_to_tuple</code>):**
    * A helper function to convert **various data types** into `Tuple` iterators, supporting:
        * `None` → `[None]`
        * `Table` → `as_tuples()`
        * `pandas.DataFrame` → Converted into a `Table`, then to Tuples
        * `List[TupleLike]` → Converted to `Tuple` instances
        * A single `TupleLike` or `Tuple` → Wrapped in an iterator

#### **Relation to <code>Tuple</code>:**

* `Table` **stores multiple <code>Tuple</code> objects** and ensures schema consistency across rows.
* Provides an **efficient bridge** between `Tuple`-based data and `pandas.DataFrame`, enabling compatibility with Python's data analysis tools.
"""

# https://github.com/Texera/texera/blob/42d803310c180978a9f02992f0e05556796b293c/core/amber/src/main/python/core/models/operator.py
operator_documentation = """### **Operator Class Overview**

The `Operator` class is an **abstract base class (ABC)** for all operators, defining the fundamental structure for processing `Tuple`, `Batch`, and `Table` data in a workflow.

#### **Key Features & Hierarchy**

1. **Base <code>Operator</code> Class**:
    * Defines lifecycle methods: `open()` and `close()`.
    * Supports a **source flag (<code>is_source</code>)** to distinguish source operators from others.
2. **Tuple-Based Processing (<code>TupleOperatorV2</code>)**:
    * Processes individual `Tuple` objects through `process_tuple(tuple_, port)`.
    * Calls `on_finish(port)` when an input port is exhausted.
3. **Types of Operators**:
    * **SourceOperator**:
        * Produces data via `produce()`, yielding `TupleLike` or `TableLike` objects.
        * Overrides `on_finish(port)` to output produced data.
    * **BatchOperator**:
        * Collects tuples into batches (`BATCH_SIZE`) before processing via `process_batch(batch, port)`.
        * Converts processed batches (typically `pandas.DataFrame`) into `Tuple` output.
    * **TableOperator**:
        * Collects tuples into a `Table` before processing via `process_table(table, port)`.
        * Converts processed `Table` output back into tuples.
4. **Data Flow & Processing**:
    * Operators receive data **tuple-by-tuple**, **batch-by-batch**, or **table-by-table** depending on the type.
    * Results are **iterators** of transformed data (`TupleLike`, `BatchLike`, or `TableLike`).
5. **Deprecated <code>TupleOperator</code>**:
    * The older version of `TupleOperator` is deprecated in favor of `TupleOperatorV2`.

#### Relation to <code>Tuple</code> and <code>Table</code>

* Operators **consume and transform** `Tuple` and `Table` data within a workflow.
* **Tuple-based operators** process row-wise, while **Table operators** handle structured table transformations.
* **Source operators** initiate the data flow by generating tuples or tables."""

udf_input_port_documentation = """
Python UDF operators support multiple input and output ports, allowing a single operator to receive different types of data from various upstream operators. In the process_tuple(self, tuple_: Tuple, port: int) function in ProcessTupleOperator and the process_table(self, table: Table, port: int) function in ProcessTableOperator, the port parameter indicates the input port. The port numbers are assigned in order, starting from 0 to N, from top to bottom. When input data have different schemas, it is necessary to assign them to different input ports. However, if all input data share the same schema, additional ports are not required. In both ProcessTupleOperator and ProcessTableOperator, there is an on_finish(self, port: int) function that is executed only after all the tuples from the specified port are processed.

Using this knowledge, for situations where multiple upstream UDFs act as input to a single UDF, we can introduce an intermediary UDF that collects all of the input data and reformats it into a single table, which is then passed as input to the original next downstream UDF. When it is necessary for this to occur in your translation from notebook to UDFs, include the intermediary UDF and make sure that it and the next operator that uses its output is formatted correctly and handles the data transfer properly.
"""

example_of_good_conversion = """
Here is an example of python code translated into a compatible Texera UDF that gives output that abides the output schema compatible with the Texera workflow operators for tuples. Other operators do not always follow this strict format, but the yielding output structure is important.

Python Code (high level idea): We have a python code that given some data, we limit the number of data.

Texera Operator code: 
from pytexera import *

class ProcessTupleOperator(UDFOperatorV2):
    def __init__(self):
        self.limit = 10
        self.count = 0
    @overrides
    def process_tuple(self, tuple_: Tuple, port: int) -> Iterator[Optional[TupleLike]]:
        if(self.count < self.limit):
            self.count += 1
            yield tuple_

"""

visualizer_documentation = """
Texera requires a unique way of generating visualizations from ML libraries:
1. Ensures one yield per operator (per Texera’s UDF constraints).
2. Uses Plotly for visualization and outputs results as embeddable HTML.
3. Error handling is built-in to notify users when data is missing.
"""

example_of_multiple_udf_conversion = """
Here is an example of breaking up python code into multiple Texera UDFs. Format your response structure exactly like the given example. The "code" key contains a dictionary of the UDF ID's with their respective code. The "edges" key contains a list of pairs that contains the connections between UDFs. The "outputs" key contains a dictionary of the UDF ID's with a list of variable names that they yield in the UDF code. The UDFs can branch and merge, it does not have to be a linear chain depending on your implementation.

Original Code:
```python
# START CELL1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# END CELL1

# START CELL2
# Load the dataset
file_path = 'diabetes.csv'
data = pd.read_csv(file_path)
# END CELL2

# START CELL3
# Remove duplicate rows
data = data.drop_duplicates()

# Remove rows with null values
data = data.dropna()
# END CELL3

# START CELL4
# Print the minimum, maximum, and mean for all fields
print("Minimum values:\n", data.min())
print("\nMaximum values:\n", data.max())
print("\nMean values:\n", data.mean())
# END CELL 4

# START CELL5
# Create a boxplot for the 'Pregnancies' field
plt.figure(figsize=(8, 6))
plt.boxplot(data['Pregnancies'], vert=False, patch_artist=True)
plt.title('Boxplot of Pregnancies')
plt.xlabel('Number of Pregnancies')
plt.show()
# END CELL5

# START CELL6
# Separate features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']
# END CELL6

# START CELL7
# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# END CELL7

# START CELL8
# Train Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.2%}")
# END CELL8

# START CELL9
# Train SVM model
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print(f"SVM Accuracy: {svm_accuracy:.2%}")
# END CELL9
```

Texera UDF conversion:
```json
{
    "code": {
        "UDF1": "# UDF1\nfrom pytexera import *\nimport pandas as pd\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        # Remove duplicate rows\n        data = table.drop_duplicates()\n\n        # Remove rows with null values\n        data = data.dropna()\n\n        # Calculate statistics\n        min_values = data.min()\n        max_values = data.max()\n        mean_values = data.mean()\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'min_values': [min_values],\n            'max_values': [max_values],\n            'mean_values': [mean_values],\n            'data': [data]\n        })\n\n        yield Table(result_table)",
        "UDF2": "# UDF2\nfrom pytexera import *\nimport pandas as pd\nimport plotly.express as px\nimport plotly.io\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n    def render_error(self, error_msg):\n        return '''<h1>Boxplot is not available.</h1>\n                  <p>Reason is: {} </p>\n               '''.format(error_msg)\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        data = table['data'].iloc[0]\n\n        if data.empty:\n            yield {'html-content': self.render_error('input table is empty.')}\n            return\n\n        # Create a boxplot for the 'Pregnancies' field\n        fig = px.box(data, x='Pregnancies')\n        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))\n\n        # Convert fig to HTML content\n        html = plotly.io.to_html(fig, include_plotlyjs='cdn', auto_play=False)\n        yield {'html-content': html}",
        "UDF3": "# UDF3\nfrom pytexera import *\nimport pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        data = table['data'].iloc[0]\n\n        # Separate features and target variable\n        X = data.drop('Outcome', axis=1)\n        y = data['Outcome']\n\n        # Split data into training and testing sets (80% train, 20% test)\n        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n        scaler = StandardScaler()\n        X_train = scaler.fit_transform(X_train)\n        X_test = scaler.transform(X_test)\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'X_train': [X_train], 'X_test': [X_test],\n            'y_train': [y_train], 'y_test': [y_test]\n        })\n\n        yield Table(result_table)",
        "UDF4": "# UDF4\nfrom pytexera import *\nimport pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import accuracy_score\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        X_train = table['X_train'].iloc[0]\n        y_train = table['y_train'].iloc[0]\n        X_test = table['X_test'].iloc[0]\n        y_test = table['y_test'].iloc[0]\n\n        # Train Random Forest model\n        rf_model = RandomForestClassifier(random_state=42)\n        rf_model.fit(X_train, y_train)\n        rf_pred = rf_model.predict(X_test)\n        rf_accuracy = accuracy_score(y_test, rf_pred)\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'rf_model': [rf_model],\n            'rf_accuracy': [rf_accuracy],\n            'X_test': [X_test],\n            'y_test': [y_test]\n        })\n\n        yield Table(result_table)",
        "UDF5": "# UDF5\nfrom pytexera import *\nimport pandas as pd\nfrom sklearn.svm import SVC\nfrom sklearn.metrics import accuracy_score\nfrom typing import Iterator, Optional\n\nclass ProcessTableOperator(UDFTableOperator):\n\n    @overrides\n    def process_table(self, table: Table, port: int) -> Iterator[Optional[TableLike]]:\n        X_train = table['X_train'].iloc[0]\n        y_train = table['y_train'].iloc[0]\n        X_test = table['X_test'].iloc[0]\n        y_test = table['y_test'].iloc[0]\n\n        # Train SVM model\n        svm_model = SVC(random_state=42)\n        svm_model.fit(X_train, y_train)\n        svm_pred = svm_model.predict(X_test)\n        svm_accuracy = accuracy_score(y_test, svm_pred)\n\n        # Create a DataFrame to yield\n        result_table = pd.DataFrame({\n            'svm_model': [svm_model],\n            'svm_accuracy': [svm_accuracy],\n            'X_test': [X_test],\n            'y_test': [y_test]\n        })\n\n        yield Table(result_table)"
    },
    "edges": [
        ["UDF1", "UDF2"],
        ["UDF1", "UDF3"],
        ["UDF3", "UDF4"],
        ["UDF3", "UDF5"]
    ],
    "outputs": {
        "UDF1": ["min_values", "max_values", "mean_values", "data"],
        "UDF2": ["html-content"],
        "UDF3": ["X_train", "X_test", "y_train", "y_test"],
        "UDF4": ["rf_model", "rf_accuracy", "X_test", "y_test"],
        "UDF5": ["svm_model", "svm_accuracy", "X_test", "y_test"]
    }
}
```
"""

example_of_mapping = """
Here is an example of a mapping generated between the given example Python code and the Texera UDFs using their CELL and UDF IDs. Cell IDs are designated by the UUID following '# START'. The format should be kept the same.
{
    "UDF1": [
        "CEll3",
        "CELL4"
    ],
    "UDF2": [
        "CELL5"
    ],
    "UDF3": [
        "CELL6",
        "CELL7"
    ]
    "UDF4": [
        "CELL8"
    ]
}
"""


openai.api_key = os.getenv("OPENAI_API_KEY")
assistant = openai.beta.assistants.create(
    name="Notebook to Texera Migrator",
    instructions=f"{texera_overview}\n"
                 f"{tuple_documentation}\n"
                 f"{table_documentation}\n"
                 f"{operator_documentation}\n"
                 f"{example_of_good_conversion}\n"
                 f"{visualizer_documentation}\n"
                 f"{udf_input_port_documentation}\n"
                 f"{example_of_multiple_udf_conversion}",
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
        sleep(1)

    messages = openai.beta.threads.messages.list(thread_id=thread_id)
    # Get the most recent assistant message
    for message in messages.data:
        if message.role == "assistant":
            return message.content[0].text.value


def python_script_to_texera(python_script: str, do_print: bool) -> str:
    starter_prompt = ("You are an expert in Python coding and workflow systems. "
                      "Many users of Texera system are non-technical, but the notebooks they provide are written by technical people. "
                      "They want to convert their notebooks to Texera workflows. "
                      "Your goal is to help convert these notebooks into a Texera workflow that non-technical users can use directly. "
                      "So do not remove or modify any classes or functions, preserve their names and structure as they are. "
                      "Ensure that all essential logic remains intact."
                      "Create multiple Texera UDF codes using the provided Python code."
                      "Number each UDF, starting at 1 and incrementing, by starting with a comment that states that UDF number.")
    using_only_class = ('Use the class and function names as shown in ProcessTupleOperator, '
                        'ProcessTableOperator, and ProcessBatchOperator. Do not change the class names, '
                        'function names, or input parameters. Use the ones that make sense and split the code '
                        'meaningfully as instructed.')
    use_starter_code = 'Use the starter code provided for Python UDFs.'
    use_documentation = ('Use the documentation of Table, Tuple, or Batch to work with parameters within Texera UDF. '
                         'Do not import other libraries to define these types.')
    no_need_for_init = ('There is no need for an __init__ function. Assume all inputs are valid pandas DataFrames, '
                        'so do not use .to_pandas(), .to_dataframe(), etc. Do not load data from a file in the first UDF, assume '
                        'that the data is already given to you in the table parameter. '
                        'Ensure proper data flow between functions. Separate operators as if they will run in different files.')
    one_output_per_operator = ('Current UDF operators can only have one output. Build a dataframe to yield all necessary variables '
                               'and data. Ensure proper data flow for each UDF and all information is yielded (including training '
                               'and testing data) if subsequent UDFs need them.')
    ensure_imports = 'Ensure all necessary imports are included in each UDF code block.'
    separate_scripts = ('Each UDF operator should be in its own Python code block. Do not combine them into a single block. '
                        'Ensure import statements cover all used functions and separate them as necessary.')
    include_all_code = ("It is VERY important that all of the original code in the Jupyter notebook is represented in the generated workflow. Make sure that nothing in the original is removed and that the semantic meaning of what the original code was doing is retained."
                        "If there are user-defined Python classes, include the entire class definition in the appriopriate UDF(s) that use that class. Always include the code that defines the class inside of every distinct UDF that uses that constructs an object of that class. Python classes are allowed in Texera UDFs and follow the same semantics as standard Python. They can be defined outside of ProcessTableOperator, ProcessTupleOperator, and ProcessBatchOperator.")
    python_to_convert = ("Return only the JSON formatted response, do not give any explanation. "
                         "Make sure the response is a valid JSON structure, including closing all braces and not including commas after the last element. "
                         "Follow this JSON format (don't reuse the values, this is just the format). 'code', 'edges', and 'outputs' are all their own key's, do not nest any of these in another one and make sure to close their braces: "
                         "{\n    \"code\": {\n        \"UDF1\": \"code for UDF1 goes here\",\n        \"UDF2\": \"code for UDF2 goes here\"\n    },\n    \"edges\": [\n        [\"UDF1\", \"UDF2\"]\n    ],\n    \"outputs\": {\n        \"UDF1\": [\"min_values\", \"max_values\", \"mean_values\", \"data\"],\n        \"UDF2\": [\"html-content\"]\n    }\n}"
                         "\nMake sure only the keys in the code section appear in the edges and outputs sections. Do not include any extraneous fields. "
                         "Do not include any extraneous UDF's in the code field that include empty strings."
                         "Give ALL of the code, do not omit anything or use placeholders for code. Make sure ALL code in the original is translated over. "
                         "Use only unescaped single quotes inside of the code values for the UDF's, do not use escaped double quotes. "
                         f"Convert following the instructions and examples given. Here is the code:\n\n{python_script}")

    final_prompt = f"{starter_prompt}\n{using_only_class}\n{use_starter_code}\n{use_documentation}\n{no_need_for_init}\n{one_output_per_operator}\n{ensure_imports}\n{separate_scripts}\n{include_all_code}\n{python_to_convert}"

    # Get workflow
    if do_print:
        print("Getting workflow from OpenAI Assistant API...", end="")
    response = call_gpt_assistant(final_prompt)
    if do_print:
        print("DONE")

    # Get mapping
    if do_print:
        print("Getting mapping from OpenAI Assistant API...", end="")
    mapping_prompt = f"{example_of_mapping}\nNow create a mapping for the UDFs and the original code. Link the code blocks marked by 'START CELL#' and 'END CELL#' with the numbered UDFs. The code between them should be equivalent. Multiple cells can be mapped to the same UDF if the code they contain are the same. There could be any number of cells and UDFs, so only create the correct number in the mapping. Only give the mapping."
    mapping = call_gpt_assistant(mapping_prompt)
    if do_print:
        print("DONE")

    return response, mapping


def parse_workflow_and_mapping(open_ai_workflow, open_ai_mapping, do_print=False):
    """
    Converts the OpenAI workflow JSON into a Texera-compatible JSON and uni-directional mapping to bidirectional mapping
    :param open_ai_workflow: OpenAI workflow
    :param open_ai_mapping: uni-directional mapping
    :return: Texera JSON and bidirectional mapping
    """
    udf_open_ai_response = json.loads(open_ai_workflow.strip("```json").strip("```").strip(), strict=False)

    workflow_json = \
        {
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


@app.route('/get_openai_response', methods=['POST'])
def get_openai_response(do_print=True):
    """
    Calls OpenAI with Texera documentation to generate and parse a workflow JSON and mapping JSON
    :param do_print: flag that controls whether the program prints which step it is on
    :return: a JSON representing the workflow, and a JSON representing the mapping
    """

    # Load the notebook
    while not NOTEBOOK_SAVED:
        sleep(2)
    notebook_path = os.path.join(NOTEBOOK_PATH, NOTEBOOK_NAME)
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

    final_workflow, final_mapping = parse_workflow_and_mapping(open_ai_workflow, open_ai_mapping, do_print)

    if do_print:
        print(final_workflow)
        print("-------------")
        print(final_mapping)

    return jsonify({"workflow": final_workflow, "mapping": final_mapping}), 200


@app.route('/set_notebook', methods=['POST'])
def set_notebook():
    try:
        global NOTEBOOK_NAME
        data = request.json
        NOTEBOOK_NAME = data.get('notebookName', 'example.ipynb')
        notebook_data = data.get('notebookData')

        for cell in notebook_data["cells"]:
            if 'metadata' not in cell:
                cell['metadata'] = {}
            cell['metadata']['uuid'] = str(uuid4())

        if not notebook_data:
            return jsonify({"error": "Notebook data is required"}), 400

        # Save the notebook JSON file
        notebook_file_path = os.path.join(NOTEBOOK_PATH, NOTEBOOK_NAME)
        with open(notebook_file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=4)
    
        global NOTEBOOK_SAVED
        NOTEBOOK_SAVED = True

        return {
            "message": "Notebook saved successfully",
            "notebookPath": notebook_file_path
        }

    except Exception as e:
        print(f"Unexpected server error: {e}")
        return {"error": f"Unexpected server error: {str(e)}"}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
