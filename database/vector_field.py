"""
Custom VectorField for pgvector support in Peewee ORM.
"""
from peewee import Field
from playhouse.postgres_ext import PostgresqlExtDatabase


class VectorField(Field):
    """
    Custom field for pgvector vector type.
    
    Usage:
        pooled_embedding = VectorField(dimensions=1152, null=True)
    """
    
    field_type = 'vector'
    
    def __init__(self, dimensions=None, *args, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)
    
    def db_value(self, value):
        """Convert Python list/numpy array to pgvector format."""
        if value is None:
            return None
        
        import numpy as np
        
        # Convert to numpy array if needed
        if isinstance(value, (list, tuple)):
            value = np.array(value, dtype=np.float32)
        elif isinstance(value, np.ndarray):
            value = value.astype(np.float32)
        
        # Convert to list for pgvector
        if isinstance(value, np.ndarray):
            value = value.tolist()
        
        # Return as string format: '[1.0,2.0,3.0]'
        return str(value).replace(' ', '')
    
    def python_value(self, value):
        """Convert pgvector format to Python list."""
        if value is None:
            return None
        
        import numpy as np
        
        # If value is already a list/array, return as is
        if isinstance(value, (list, np.ndarray)):
            return np.array(value, dtype=np.float32) if isinstance(value, list) else value
        
        # If value is a string, parse it
        if isinstance(value, str):
            # Remove brackets and split
            value = value.strip('[]')
            return np.array([float(x) for x in value.split(',')], dtype=np.float32)
        
        return value
    
    def get_column_type(self):
        """Return PostgreSQL column type."""
        if self.dimensions:
            return f'vector({self.dimensions})'
        return 'vector'
