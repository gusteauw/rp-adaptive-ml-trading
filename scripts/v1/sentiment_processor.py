import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Optional
from pathlib import Path
import torch
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import json

class SentimentProcessor:
    """Process and aggregate sentiment data from multiple sources"""
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model and tokenizer
        self.logger.info(f"Loading {model_name} model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        
        # Create necessary directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            "data/sentiment/processed",
            "data/sentiment/aggregated",
            "data/sentiment/metadata"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def process_batch(self, texts: List[str]) -> np.ndarray:
        """Process a batch of texts through FinBERT"""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
        return probs.cpu().numpy()
    
    def process_texts(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Process all texts and return sentiment scores"""
        self.logger.info(f"Processing {len(texts)} texts...")
        
        all_probs = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            batch_probs = self.process_batch(batch_texts)
            all_probs.append(batch_probs)
        
        all_probs = np.vstack(all_probs)
        
        return {
            'negative': all_probs[:, 0],
            'neutral': all_probs[:, 1],
            'positive': all_probs[:, 2]
        }
    
    def aggregate_daily_sentiment(
        self,
        df: pd.DataFrame,
        text_column: str,
        date_column: str,
        ticker_column: Optional[str] = None
    ) -> pd.DataFrame:
        """Aggregate sentiment scores to daily level"""
        # Process all texts
        sentiments = self.process_texts(df[text_column].tolist())
        
        # Add sentiment scores to dataframe
        for sentiment, scores in sentiments.items():
            df[f'sentiment_{sentiment}'] = scores
        
        # Calculate compound score
        df['sentiment_compound'] = (
            df['sentiment_positive'] - df['sentiment_negative']
        )
        
        # Group by date and ticker if provided
        group_cols = [date_column]
        if ticker_column:
            group_cols.append(ticker_column)
        
        agg_dict = {
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean'
        }
        
        daily_sentiment = df.groupby(group_cols).agg(agg_dict)
        daily_sentiment.columns = [
            f"{col[0]}_{col[1]}" for col in daily_sentiment.columns
        ]
        
        return daily_sentiment.reset_index()
    
    def calculate_sentiment_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional sentiment metrics"""
        # Sentiment momentum (5-day moving average)
        df['sentiment_momentum'] = df.groupby('ticker')['sentiment_compound_mean'].transform(
            lambda x: x.rolling(window=5).mean()
        )
        
        # Sentiment volatility
        df['sentiment_volatility'] = df.groupby('ticker')['sentiment_compound_mean'].transform(
            lambda x: x.rolling(window=10).std()
        )
        
        # Sentiment regime (based on moving average crossover)
        df['sentiment_ma_fast'] = df.groupby('ticker')['sentiment_compound_mean'].transform(
            lambda x: x.rolling(window=5).mean()
        )
        df['sentiment_ma_slow'] = df.groupby('ticker')['sentiment_compound_mean'].transform(
            lambda x: x.rolling(window=20).mean()
        )
        df['sentiment_regime'] = (df['sentiment_ma_fast'] > df['sentiment_ma_slow']).astype(int)
        
        # Sentiment volume impact
        df['sentiment_impact'] = df['sentiment_compound_mean'] * df['sentiment_compound_count']
        
        return df
    
    def process_and_save(
        self,
        input_path: str,
        output_path: str,
        text_column: str,
        date_column: str,
        ticker_column: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Main processing pipeline"""
        # Load data
        self.logger.info(f"Loading data from {input_path}...")
        df = pd.read_csv(input_path, parse_dates=[date_column])
        
        # Aggregate daily sentiment
        daily_sentiment = self.aggregate_daily_sentiment(
            df, text_column, date_column, ticker_column
        )
        
        # Calculate additional metrics
        if ticker_column:
            daily_sentiment = self.calculate_sentiment_metrics(daily_sentiment)
        
        # Save processed data
        self.logger.info(f"Saving processed data to {output_path}...")
        daily_sentiment.to_csv(output_path, index=False)
        
        # Save metadata if provided
        if metadata:
            metadata_path = Path(output_path).parent / "metadata" / f"{Path(output_path).stem}_meta.json"
            metadata.update({
                'processed_date': datetime.now().isoformat(),
                'model_name': self.model_name,
                'n_records': len(df),
                'date_range': [
                    df[date_column].min().isoformat(),
                    df[date_column].max().isoformat()
                ]
            })
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    # Example usage
    processor = SentimentProcessor()
    
    # Process news sentiment
    processor.process_and_save(
        input_path="data/raw/news_data.csv",
        output_path="data/sentiment/processed/news_sentiment.csv",
        text_column="headline",
        date_column="date",
        ticker_column="ticker",
        metadata={'source': 'financial_news'}
    )
    
    # Process social media sentiment
    processor.process_and_save(
        input_path="data/raw/social_data.csv",
        output_path="data/sentiment/processed/social_sentiment.csv",
        text_column="text",
        date_column="date",
        ticker_column="ticker",
        metadata={'source': 'social_media'}
    ) 