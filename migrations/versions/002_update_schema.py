"""Update schema

Revision ID: 002_update_schema
Revises: 001_initial_schema
Create Date: 2025-01-08
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = '002_update_schema'
down_revision = '001_initial_schema'
branch_labels = None
depends_on = None

def table_exists(table_name):
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    return table_name in inspector.get_table_names()

def column_exists(table_name, column_name):
    conn = op.get_bind()
    inspector = Inspector.from_engine(conn)
    columns = [c['name'] for c in inspector.get_columns(table_name)]
    return column_name in columns

def upgrade() -> None:
    # Add new columns to trained_model table if they don't exist
    if not column_exists('trained_model', 'model_type'):
        op.add_column('trained_model', sa.Column('model_type', sa.String(), nullable=False, server_default='lstm'))
    
    if not column_exists('trained_model', 'is_active'):
        op.add_column('trained_model', sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'))
    
    # Add confidence_score to prediction table if it doesn't exist
    if not column_exists('prediction', 'confidence_score'):
        op.add_column('prediction', sa.Column('confidence_score', sa.Float(), nullable=True))
    
    # Drop old tables only if they exist
    tables_to_drop = ['sensor_readings', 'predictions', 'model_metadata', 'trained_models']
    
    for table in tables_to_drop:
        if table_exists(table):
            op.drop_table(table)

def downgrade() -> None:
    # Remove new columns if they exist
    if column_exists('trained_model', 'model_type'):
        op.drop_column('trained_model', 'model_type')
    
    if column_exists('trained_model', 'is_active'):
        op.drop_column('trained_model', 'is_active')
    
    if column_exists('prediction', 'confidence_score'):
        op.drop_column('prediction', 'confidence_score')
    
    # Only recreate tables if they don't exist
    if not table_exists('sensor_readings'):
        op.create_table('sensor_readings',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('sensor_id', sa.String(), nullable=True),
            sa.Column('temperature', sa.Float(), nullable=True),
            sa.Column('humidity', sa.Float(), nullable=True),
            sa.Column('timestamp', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )
    
    if not table_exists('predictions'):
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
    
    if not table_exists('model_metadata'):
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
    
    if not table_exists('trained_models'):
        op.create_table('trained_models',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('sensor_id', sa.String(), nullable=True),
            sa.Column('model_data', sa.LargeBinary(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id')
        )