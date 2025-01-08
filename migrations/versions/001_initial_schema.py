"""Initial schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2025-01-08
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001_initial_schema'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Create all tables from the existing create_prediction_tables.py
    op.create_table(
        'sensor',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('location', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table(
        'trained_model',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sensor_id', sa.Integer(), nullable=False),
        sa.Column('model_data', sa.LargeBinary(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.Column('model_metrics', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['sensor_id'], ['sensor.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_table(
        'prediction',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sensor_id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('prediction_value', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['model_id'], ['trained_model.id'], ),
        sa.ForeignKeyConstraint(['sensor_id'], ['sensor.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    op.drop_table('prediction')
    op.drop_table('trained_model')
    op.drop_table('sensor')