"""Datasets and immutable analysis runs."""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("datasets", sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False), sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("data", sa.Text(), nullable=False), sa.Column("profile", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_datasets_digest", "datasets", ["digest"], unique=True)
    op.create_table("analysis_runs", sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("dataset_id", sa.String(36), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("result", sa.JSON(), nullable=False), sa.Column("artifact", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.create_index("ix_analysis_runs_dataset_id", "analysis_runs", ["dataset_id"])


def downgrade():
    op.drop_table("analysis_runs")
    op.drop_table("datasets")
