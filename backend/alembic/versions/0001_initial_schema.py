"""initial schema: pgvector + core tables

Revision ID: 0001
Revises:
"""
from collections.abc import Sequence
import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 768


def _ts(name: str) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "applications",
        sa.Column("id", sa.String(50), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("description", sa.Text),
        _ts("created_at"), _ts("updated_at"),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("name", sa.String(200)),
        _ts("created_at"),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("application_id", sa.String(50),
                  sa.ForeignKey("applications.id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("file_type", sa.String(20), nullable=False),
        sa.Column("storage_path", sa.String(500), nullable=False),
        sa.Column("file_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.Enum("PENDING", "PROCESSING", "READY", "FAILED",
                                    name="document_status"), nullable=False),
        sa.Column("error_message", sa.Text),
        _ts("created_at"), _ts("updated_at"),
        sa.UniqueConstraint("application_id", "file_hash", name="uq_documents_app_hash"),
    )
    op.create_index("ix_documents_application_id", "documents", ["application_id"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("document_id", sa.Uuid,
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("application_id", sa.String(50),
                  sa.ForeignKey("applications.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("page", sa.Integer),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("embedding", Vector(EMBEDDING_DIM)),
        _ts("created_at"),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_application_id", "chunks", ["application_id"])
    op.create_index("ix_chunks_embedding_hnsw", "chunks", ["embedding"],
                    postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"})

    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("user_id", sa.Uuid, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("application_id", sa.String(50),
                  sa.ForeignKey("applications.id", ondelete="CASCADE"), nullable=False),
        _ts("created_at"), _ts("updated_at"),
    )
    op.create_index("ix_chat_sessions_application_id", "chat_sessions", ["application_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("session_id", sa.Uuid,
                  sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        _ts("created_at"),
    )
    op.create_index("ix_messages_session_id", "messages", ["session_id"])

    op.create_table(
        "retrieval_logs",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("session_id", sa.Uuid, sa.ForeignKey("chat_sessions.id", ondelete="CASCADE")),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("retrieved_chunk_ids", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("scores", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("latency_ms", sa.Float),
        _ts("created_at"),
    )
    op.create_index("ix_retrieval_logs_session_id", "retrieval_logs", ["session_id"])


def downgrade() -> None:
    for t in ("retrieval_logs", "messages", "chat_sessions", "chunks",
              "documents", "users", "applications"):
        op.drop_table(t)
    op.execute("DROP TYPE IF EXISTS document_status")