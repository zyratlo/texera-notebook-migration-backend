from fastapi import APIRouter, Request, HTTPException
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import Json, RealDictCursor

router = APIRouter(prefix="/postgres", tags=["postgres"])

DB_CONFIG = {
    "host": "host.docker.internal",
    "port": 5432,
    "database": "texera_db",
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
        version = data["version"]
        notebook_data = data["notebook"]

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Insert into notebook_migration_notebook_data
                cur.execute("""
                    INSERT INTO texera_db.notebook_migration_notebook_data (wid, notebook)
                    VALUES (%s, %s)
                    """, (wid, Json(notebook_data)))

                # Insert into notebook_migration_mapping_data
                cur.execute("""
                    INSERT INTO texera_db.notebook_migration_mapping_data (wid, version, mapping)
                    VALUES (%s, %s, %s)
                    """, (wid, version, Json(mapping_data)))

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
        version = data["version"]

        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        nmn.notebook,
                        nmm.mapping
                    FROM 
                        texera_db.notebook_migration_notebook_data AS nmn
                    JOIN 
                        texera_db.notebook_migration_mapping_data AS nmm
                        ON nmn.wid = nmm.wid
                    WHERE 
                        nmn.wid = %s
                        AND nmm.version = %s;
                """, (wid, version))
                result = cur.fetchone()

        if not result:
            return {"exists": False}

        return {"exists": True, "mapping": result["mapping"], "notebook": result["notebook"]}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {str(e)}")
