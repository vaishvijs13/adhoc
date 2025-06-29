from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import faiss
import os
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class EmbeddingUpdateRequest(BaseModel):
    text: str
    politician: str
    date: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[dict] = None


class EmbeddingUpdateResponse(BaseModel):
    success: bool
    message: str
    added_count: int
    total_embeddings: int


class DatabaseStats(BaseModel):
    total_embeddings: int
    unique_politicians: int
    latest_update: str
    database_size_mb: float
    index_type: str


class EmbeddingUpdater:
    def __init__(
        self,
        embeddings_file="data/reference_embeddings.csv",
        index_file="data/reference_index.faiss",
    ):
        self.embeddings_file = embeddings_file
        self.index_file = index_file
        self.backup_dir = "data/backups"
        os.makedirs(self.backup_dir, exist_ok=True)

    def create_backup(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if os.path.exists(self.embeddings_file):
            backup_csv = f"{self.backup_dir}/embeddings_backup_{timestamp}.csv"
            shutil.copy2(self.embeddings_file, backup_csv)

        if os.path.exists(self.index_file):
            backup_faiss = f"{self.backup_dir}/index_backup_{timestamp}.faiss"
            shutil.copy2(self.index_file, backup_faiss)

        return timestamp

    def add_embedding(
        self,
        text: str,
        politician: str,
        embedding: np.ndarray,
        date: str = None,
        source: str = None,
        metadata: dict = None,
    ):
        try:
            if os.path.exists(self.embeddings_file):
                df = pd.read_csv(self.embeddings_file)
            else:
                df = pd.DataFrame(
                    columns=[
                        "text",
                        "politician",
                        "date",
                        "source",
                        "metadata",
                        "embedding",
                    ]
                )

            # create new row
            new_row = {
                "text": text,
                "politician": politician,
                "date": date or datetime.now().isoformat(),
                "source": source or "manual_upload",
                "metadata": str(metadata) if metadata else "",
                "embedding": ",".join(map(str, embedding.flatten())),
            }

            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            df.to_csv(self.embeddings_file, index=False)

            # rebuild FAISS idx
            self._rebuild_faiss_index(df)

            return True

        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            return False

    def _rebuild_faiss_index(self, df):
        try:
            embeddings = []
            for embedding_str in df["embedding"]:
                embedding = np.array([float(x) for x in embedding_str.split(",")])
                embeddings.append(embedding)

            embeddings = np.array(embeddings).astype("float32")

            # create new FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # save index
            faiss.write_index(index, self.index_file)

        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}")
            raise

    def get_stats(self):
        try:
            if not os.path.exists(self.embeddings_file):
                return DatabaseStats(
                    total_embeddings=0,
                    unique_politicians=0,
                    latest_update="Never",
                    database_size_mb=0.0,
                    index_type="None",
                )

            df = pd.read_csv(self.embeddings_file)

            # calculate file sizes
            csv_size = (
                os.path.getsize(self.embeddings_file)
                if os.path.exists(self.embeddings_file)
                else 0
            )
            faiss_size = (
                os.path.getsize(self.index_file)
                if os.path.exists(self.index_file)
                else 0
            )
            total_size_mb = (csv_size + faiss_size) / (1024 * 1024)

            latest_update = df["date"].max() if not df.empty else "Never"

            return DatabaseStats(
                total_embeddings=len(df),
                unique_politicians=df["politician"].nunique() if not df.empty else 0,
                latest_update=latest_update,
                database_size_mb=round(total_size_mb, 2),
                index_type="FAISS IndexFlatL2",
            )

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error getting database stats: {e}"
            )


# global updater instance
updater = EmbeddingUpdater()


@router.post("/add-statement", response_model=EmbeddingUpdateResponse)
async def add_political_statement(request: EmbeddingUpdateRequest):
    try:
        from models import get_model

        model = get_model()
        if model is None:
            raise HTTPException(status_code=500, detail="Model not initialized")

        import torch
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model.embedding_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        # create backup before adding
        backup_timestamp = updater.create_backup()

        # add embedding to database
        success = updater.add_embedding(
            text=request.text,
            politician=request.politician,
            embedding=embedding,
            date=request.date,
            source=request.source,
            metadata=request.metadata,
        )

        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to add embedding to database"
            )

        stats = updater.get_stats()

        return EmbeddingUpdateResponse(
            success=True,
            message=f"Successfully added statement for {request.politician}. Backup created: {backup_timestamp}",
            added_count=1,
            total_embeddings=stats.total_embeddings,
        )

    except Exception as e:
        logger.error(f"Error adding political statement: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding statement: {e}")


@router.post("/bulk-upload", response_model=EmbeddingUpdateResponse)
async def bulk_upload_statements(file: UploadFile = File(...)):
    try:
        content = await file.read()

        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(content)

        df = pd.read_csv(temp_file)
        required_columns = ["text", "politician"]

        if not all(col in df.columns for col in required_columns):
            os.remove(temp_file)
            raise HTTPException(
                status_code=400, detail=f"CSV must contain columns: {required_columns}"
            )

        from models import get_model

        model = get_model()
        if model is None:
            os.remove(temp_file)
            raise HTTPException(status_code=500, detail="Model not initialized")

        backup_timestamp = updater.create_backup()

        added_count = 0
        for _, row in df.iterrows():
            try:
                # generate embedding
                import torch
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                inputs = tokenizer(
                    row["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                with torch.no_grad():
                    outputs = model.embedding_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

                # add to database
                success = updater.add_embedding(
                    text=row["text"],
                    politician=row["politician"],
                    embedding=embedding,
                    date=row.get("date"),
                    source=row.get("source", "bulk_upload"),
                    metadata={"original_row": row.to_dict()},
                )

                if success:
                    added_count += 1

            except Exception as e:
                logger.warning(f"Failed to process row: {e}")
                continue

        os.remove(temp_file)

        stats = updater.get_stats()

        return EmbeddingUpdateResponse(
            success=True,
            message=f"Bulk upload completed. Added {added_count} statements. Backup: {backup_timestamp}",
            added_count=added_count,
            total_embeddings=stats.total_embeddings,
        )

    except Exception as e:
        logger.error(f"Error in bulk upload: {e}")
        if "temp_file" in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {e}")


@router.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    return updater.get_stats()


@router.post("/rebuild-index")
async def rebuild_faiss_index():
    try:
        if not os.path.exists(updater.embeddings_file):
            raise HTTPException(status_code=404, detail="No embeddings database found")

        backup_timestamp = updater.create_backup()

        df = pd.read_csv(updater.embeddings_file)
        updater._rebuild_faiss_index(df)

        return {
            "success": True,
            "message": f"FAISS index rebuilt successfully. Backup: {backup_timestamp}",
            "embeddings_count": len(df),
        }

    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {e}")


@router.delete("/clear-database")
async def clear_database():
    try:
        backup_timestamp = updater.create_backup()

        if os.path.exists(updater.embeddings_file):
            os.remove(updater.embeddings_file)
        if os.path.exists(updater.index_file):
            os.remove(updater.index_file)

        return {
            "success": True,
            "message": f"Database cleared. Backup created: {backup_timestamp}",
            "backup_location": f"{updater.backup_dir}/embeddings_backup_{backup_timestamp}.csv",
        }

    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")
