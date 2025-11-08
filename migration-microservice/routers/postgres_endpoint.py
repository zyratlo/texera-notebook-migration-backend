from fastapi import APIRouter, Request, HTTPException
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import Json, RealDictCursor

router = APIRouter(prefix="/postgres", tags=["postgres"])

DB_CONFIG = {
    "host": "host.docker.internal",
    "port": 5432,
    "database": "texera_notebook_migration_db",
    "user": "texera",
    "password": "password"
}


@contextmanager
def get_db_connection():
    """Context manager for safe Postgres connection handling."""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Postgres database connection error: {str(e)}")
    finally:
        if conn:
            conn.close()


@router.post("/store_mapping_and_workflow")
async def store_mapping_and_workflow(request: Request):
    try:
        data = await request.json()
        wid = data["wid"]
        mapping_data = data["mapping"]
        notebook_data = data["notebook"]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO texera_notebook_migration_db.migration_data (wid, mapping, notebook)
                    VALUES (%s, %s, %s)
                    """, (wid, Json(mapping_data), Json(notebook_data)))
                conn.commit()

        return {"message": "Mapping and workflow stored successfully."}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")


@router.post("/get_mapping_and_workflow")
async def get_mapping_and_workflow(request: Request):
    try:
        data = await request.json()
        wid = data["wid"]

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT mapping, notebook
                    FROM texera_notebook_migration_db.migration_data
                    WHERE wid = %s;
                """, (wid,))
                result = cur.fetchone()

        if not result:
            return {"exists": False}

        return {"exists": True, "mapping": result["mapping"], "notebook": result["notebook"]}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")


@router.post("/delete_by_wid")
async def delete_by_wid(request: Request):
    try:
        data = await request.json()
        wid = data["wid"]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM texera_notebook_migration_db.migration_data
                    WHERE wid = %s
                    RETURNING wid;
                """, (wid,))
                deleted = cur.fetchone()
                conn.commit()

        if not deleted:
            return {"deleted": False, "message": f"No entry found for wid={wid}"}

        return {"deleted": True, "message": f"Entry with wid={wid} successfully deleted."}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
