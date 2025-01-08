"""Update schema

Revision ID: 002_update_schema
Revises: 001_initial_schema
Create Date: 2025-01-08
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '002_update_schema'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Add new columns to trained_model table
    op.add_column('trained_model', sa.Column('model_type', sa.String(), nullable=False, server_default='lstm'))
    op.add_column('trained_model', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'))
    
    # Add confidence_score to prediction table
    op.add_column('prediction', sa.Column('confidence_score', sa.Float(), nullable=True))
    
    # Drop old tables that are no longer used
    op.drop_table('sensor_readings')
    op.drop_table('predictions')
    op.drop_table('model_metadata')
    op.drop_table('trained_models')

def downgrade() -> None:
    # Remove new columns
    op.drop_column('trained_model', 'model_type')
    op.drop_column('trained_model', 'is_active')
    op.drop_column('prediction', 'confidence_score')
    
    # Recreate old tables
    op.create_table('sensor_readings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sensor_id', sa.String(), nullable=True),
        sa.Column('temperature', sa.Float(), nullable=True),
        sa.Column('humidity', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sensor_id', sa.String(), nullable=True),
        sa.Column('temperature_prediction', sa.Float(), nullable=True),
        sa.Column('humidity_prediction', sa.Float(), nullable=True),
        sa.Column('prediction_timestamp', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('model_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('version', sa.String(), nullable=True),
        sa.Column('metrics', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name')
    )
    
    op.create_table('trained_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sensor_id', sa.String(), nullable=True),
        sa.Column('model_data', sa.LargeBinary(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )