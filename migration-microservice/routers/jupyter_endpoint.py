import os
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/jupyter", tags=["jupyter"])


@router.post("/set_notebook")
async def set_notebook(request: Request):
    try:
        data = await request.json()
        notebook_name = data.get('notebookName', 'notebook.ipynb')
        notebook_data = data.get('notebookData')

        if not notebook_data:
            raise HTTPException(status_code=400, detail="Notebook data is required")

        # Save the notebook JSON file
        notebook_file_path = os.path.join("/home/jovyan/work", notebook_name)
        with open(notebook_file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook_data, f, indent=4)

        return JSONResponse(content={
            "message": "Notebook saved successfully",
            "notebookPath": notebook_file_path
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
