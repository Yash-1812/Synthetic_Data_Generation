from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

class SyntheticDataGenerator:
    def __init__(self, schema):
        self.schema = schema
        self.model = None
        self.metadata = None

    def _create_metadata(self, df):
        """Create metadata from dataframe"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        return metadata

    def train(self, df):
        """Train CTGAN model"""
        self.metadata = self._create_metadata(df)

        self.model = CTGANSynthesizer(self.metadata)
        self.model.fit(df)

    def generate(self, num_rows):
        """Generate synthetic data"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        return self.model.sample(num_rows)