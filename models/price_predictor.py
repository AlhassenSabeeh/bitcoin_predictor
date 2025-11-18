import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier
import pickle
import os
import json
from sklearn.metrics import precision_score, classification_report
from datetime import datetime, timedelta


class BitcoinPricePredictor:
    def __init__(self):
        self.model = None
        self.predictors = []
        self.feature_names = []
        self.training_date = None
        self.backtest_precision = None
        self.backtest_accuracy = None

    def load_data(self):
        """Load and prepare Bitcoin price data with robust error handling"""
        print("📊 Loading Bitcoin price data from Yahoo Finance...")

        try:
            # FIXED: Use proper date range for Bitcoin (Bitcoin started in
            # 2009)
            btc_ticker = yf.Ticker("BTC-USD")

            # Try multiple approaches to get Bitcoin data
            btc = None

            # Approach 1: Try with specific period
            try:
                btc = btc_ticker.history(period="max")
                print("✅ Loaded Bitcoin data using period='max'")
            except Exception as e:
                print(f"⚠️  Period 'max' failed: {e}")

            # Approach 2: If period fails, try with specific start date
            # (Bitcoin inception)
            if btc is None or btc.empty:
                try:
                    start_date = "2010-07-18"  # Bitcoin started around this time
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    btc = btc_ticker.history(start=start_date, end=end_date)
                    print(f"✅ Loaded Bitcoin data from {start_date} to {end_date}")
                except Exception as e:
                    print(f"⚠️  Date range failed: {e}")

            # Approach 3: Try with shorter period as fallback
            if btc is None or btc.empty:
                try:
                    btc = btc_ticker.history(period="5y")
                    print("✅ Loaded Bitcoin data using period='5y'")
                except Exception as e:
                    print(f"⚠️  Period '5y' failed: {e}")

            # If all approaches fail, create sample data
            if btc is None or btc.empty:
                print(
                    "❌ All Yahoo Finance methods failed, creating sample Bitcoin data"
                )
                return self.create_sample_bitcoin_data()

            btc = btc.reset_index()
            if "Date" in btc.columns:
                btc["Date"] = (
                    btc["Date"].dt.tz_localize(None)
                    if hasattr(btc["Date"].dt, "tz_localize")
                    else btc["Date"]
                )

            # Remove unnecessary columns if they exist
            for col in ["Dividends", "Stock Splits"]:
                if col in btc.columns:
                    del btc[col]

            # Standardize column names
            btc.columns = [c.lower() for c in btc.columns]

            print(
                f"✅ Loaded {len(btc)} days of Bitcoin data (up to {btc['date'].max()})"
            )
            return btc

        except Exception as e:
            print(f"❌ Error loading Bitcoin data: {e}")
            print("🔄 Creating sample Bitcoin data as fallback...")
            return self.create_sample_bitcoin_data()

    def create_sample_bitcoin_data(self):
        """Create realistic sample Bitcoin data when Yahoo Finance fails"""
        print("📝 Creating realistic sample Bitcoin data...")

        from datetime import datetime, timedelta
        import random

        # Create data from 2010 to present (Bitcoin timeline)
        start_date = datetime(2010, 7, 18)
        end_date = datetime.now()
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        sample_data = []
        base_price = 0.08  # Bitcoin started around $0.08

        for date in dates:
            # Simulate realistic Bitcoin price growth with volatility
            days_since_start = (date - start_date).days

            # Basic growth trend (exponential with some noise)
            if days_since_start == 0:
                price = base_price
            else:
                # Exponential growth with random fluctuations
                growth_rate = 0.0003  # Daily growth rate
                volatility = random.uniform(-0.1, 0.1)

                previous_price = sample_data[-1]["close"] if sample_data else base_price
                price = previous_price * (1 + growth_rate + volatility)

                # Ensure price doesn't go too low
                price = max(price, 0.01)

            # Generate OHLC data with some variation
            open_price = price * (1 + random.uniform(-0.02, 0.02))
            high = max(open_price, price) * (1 + random.uniform(0, 0.05))
            low = min(open_price, price) * (1 - random.uniform(0, 0.05))
            volume = random.randint(1000000, 50000000)

            sample_data.append(
                {
                    "date": date,
                    "open": round(open_price, 2),
                    "high": round(high, 2),
                    "low": round(low, 2),
                    "close": round(price, 2),
                    "volume": volume,
                }
            )

        df = pd.DataFrame(sample_data)
        print(f"✅ Created realistic sample Bitcoin data for {len(df)} days")
        return df

    def merge_with_sentiment(self, btc_data):
        """Merge price data with sentiment data - FIXED DATA TYPE ISSUE"""
        print("🔄 Merging price data with Wikipedia sentiment...")

        if btc_data.empty:
            print("❌ No Bitcoin data to merge")
            return pd.DataFrame()

        try:
            # Load sentiment data
            if not os.path.exists("wikipedia_edits.csv"):
                print("❌ Sentiment data file not found")
                # Create empty sentiment columns
                btc_data["sentiment"] = 0.0
                btc_data["neg_sentiment"] = 0.0
                btc_data["edit_count"] = 0
                return btc_data

            # FIXED: Read CSV with proper error handling
            try:
                bit_sent = pd.read_csv(
                    "wikipedia_edits.csv", index_col=0, parse_dates=True
                )
                print(f"✅ Loaded sentiment data for {len(bit_sent)} days")
            except Exception as e:
                print(f"❌ Error reading sentiment file: {e}")
                # Create empty sentiment columns as fallback
                btc_data["sentiment"] = 0.0
                btc_data["neg_sentiment"] = 0.0
                btc_data["edit_count"] = 0
                return btc_data

            # Prepare dates for merging - FIXED DATA TYPE ISSUE
            btc_data["date"] = pd.to_datetime(btc_data["date"]).dt.normalize()

            # Ensure sentiment data index is datetime - FIXED MERGE ERROR
            if not pd.api.types.is_datetime64_any_dtype(bit_sent.index):
                bit_sent.index = pd.to_datetime(bit_sent.index)
            bit_sent.index = bit_sent.index.normalize()

            # Merge datasets - FIXED: Ensure same data types
            btc_data = btc_data.merge(
                bit_sent, left_on="date", right_index=True, how="left"
            )

            # Fill missing sentiment values with proper data types
            sentiment_cols = ["sentiment", "neg_sentiment", "edit_count"]
            for col in sentiment_cols:
                if col in btc_data.columns:
                    btc_data[col] = pd.to_numeric(
                        btc_data[col], errors="coerce"
                    ).fillna(0.0)
                else:
                    if col == "edit_count":
                        btc_data[col] = 0
                    else:
                        btc_data[col] = 0.0

            btc_data = btc_data.set_index("date")
            print("✅ Successfully merged price and sentiment data")
            btc_data.to_csv("merged_price_and_sentimen.csv")
            print("✅ Successfully generte merged_data.csv")
            return btc_data

        except Exception as e:
            print(f"❌ Error merging sentiment data: {e}")
            # Add default sentiment columns
            btc_data["sentiment"] = 0.0
            btc_data["neg_sentiment"] = 0.0
            btc_data["edit_count"] = 0
            if "date" in btc_data.columns:
                btc_data = btc_data.set_index("date")
            return btc_data

    def create_features(self, data):
        """Create technical features for prediction"""
        print("⚙️  Creating technical features...")

        if data.empty:
            print("❌ No data for feature creation")
            return data

        # used the enhanced feature engineering
        data = self.compute_rolling_features(data)

        print(f"✅ Created {len(self.predictors)} enhanced features")
        return data

    def train_model(self, data):
        """Enhanced training with backtesting validation"""
        print("🤖 Training XGBoost model with backtesting...")

        if data.empty:
            print("❌ No data for training")
            return None

        if len(self.predictors) == 0:
            print("❌ No predictors available for training")
            return None

        try:
            # Ensure all predictors exist and are numeric
            for predictor in self.predictors:
                if predictor not in data.columns:
                    print(f"⚠️  Predictor {predictor} not found, adding zeros")
                    if predictor == "edit_count" or predictor.startswith("edit_"):
                        data[predictor] = 0
                    else:
                        data[predictor] = 0.0
                else:
                    # Ensure numeric type
                    data[predictor] = pd.to_numeric(
                        data[predictor], errors="coerce"
                    ).fillna(0.0)

            # STEP 1: Run backtesting for realistic performance metrics
            backtest_predictions = self.backtest(data)

            if not backtest_predictions.empty:
                precision = precision_score(
                    backtest_predictions["target"], backtest_predictions["predictions"]
                )
                accuracy = (
                    backtest_predictions["target"]
                    == backtest_predictions["predictions"]
                ).mean()

                # This is the key metric!
                print(f"🎯 BACKTEST Precision: {precision:.2%}")
                print(f"🎯 BACKTEST Accuracy: {accuracy:.2%}")

                # Store backtest results for dashboard
                self.backtest_precision = precision
                self.backtest_accuracy = accuracy
            else:
                print("⚠️  Backtesting returned no results")
                self.backtest_precision = 0.5  # Default to random
                self.backtest_accuracy = 0.5

            # STEP 2: Train final model on ALL data for production use
            self.model = XGBClassifier(
                random_state=42,
                learning_rate=0.1,
                n_estimators=100,
                eval_metric="logloss",
                verbosity=1,
            )

            print(
                f"📈 Final training on {len(data)} samples with {len(self.predictors)} features..."
            )
            self.model.fit(data[self.predictors], data["target"])

            # Store feature names and training date
            self.model._feature_names = self.predictors
            self.training_date = datetime.now()

            # Save model with enhanced info including backtest results
            os.makedirs("models/saved_models", exist_ok=True)
            model_path = "models/saved_models/bitcoin_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            feature_info = {
                "predictors": self.predictors,
                "feature_names": self.feature_names,
                "training_date": self.training_date.isoformat(),
                "training_samples": len(data),
                "backtest_precision": float(getattr(self, "backtest_precision", 0.5)),
                "backtest_accuracy": float(getattr(self, "backtest_accuracy", 0.5)),
                "data_date_range": {
                    "start": str(data.index.min())
                    if hasattr(data.index, "min")
                    else "Unknown",
                    "end": str(data.index.max())
                    if hasattr(data.index, "max")
                    else "Unknown",
                },
            }

            with open("models/saved_models/feature_info.json", "w") as f:
                json.dump(feature_info, f, indent=2)

            print("✅ Model training complete with backtesting validation")
            return self.model

        except Exception as e:
            print(f"❌ Error training model: {e}")
            import traceback

            traceback.print_exc()
            return None

    def predict_next_day(self, data):
        """Predict next day price movement"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")

        if data.empty:
            raise ValueError("No data available for prediction")

        try:
            # Get the latest data point
            latest_data = data.iloc[-1:].copy()

            # Ensure all predictors are present and numeric
            for pred in self.predictors:
                if pred not in latest_data.columns:
                    print(f"⚠️  Predictor {pred} not found, using 0")
                    if pred == "edit_count" or pred.startswith("edit_"):
                        latest_data[pred] = 0
                    else:
                        latest_data[pred] = 0.0
                else:
                    latest_data[pred] = pd.to_numeric(
                        latest_data[pred], errors="coerce"
                    ).fillna(0.0)

            prediction = self.model.predict(latest_data[self.predictors])
            prediction_proba = self.model.predict_proba(latest_data[self.predictors])

            confidence = float(np.max(prediction_proba[0]))
            current_price = float(round(latest_data["close"].iloc[0], 2))

            result = {
                "prediction": "UP" if prediction[0] == 1 else "DOWN",
                "confidence": round(confidence * 100, 2),
                "current_price": current_price,
                "prediction_proba": {
                    "up_probability": float(round(prediction_proba[0][1] * 100, 2)),
                    "down_probability": float(round(prediction_proba[0][0] * 100, 2)),
                },
                "model_training_date": self.training_date.strftime("%Y-%m-%d")
                if self.training_date
                else "Unknown",
            }

            print(
                f"🎯 Prediction: {result['prediction']} (Confidence: {result['confidence']}%)"
            )
            return result

        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            # Return a safe default prediction
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {"up_probability": 50.0, "down_probability": 50.0},
                "model_training_date": "Unknown",
                "error": str(e),
            }

    def get_feature_importance(self):
        """Get feature importance for analytics"""
        if self.model is None:
            return {"error": "Model not trained"}

        try:
            if hasattr(self.model, "feature_importances_"):
                importance_scores = self.model.feature_importances_.tolist()
                feature_names = getattr(self.model, "_feature_names", self.predictors)

                # Combine and sort by importance
                features = list(zip(feature_names, importance_scores))
                features.sort(key=lambda x: x[1], reverse=True)

                # Return top features
                top_features = features[:15]

                return {
                    "features": [f[0] for f in top_features],
                    "importance": [float(f[1]) for f in top_features],
                    "total_features": len(features),
                    "backtest_precision": float(getattr(self, 'backtest_precision', 0.5)),
                    "backtest_accuracy": float(getattr(self, 'backtest_accuracy', 0.5))
                }
            else:
                return {"error": "No feature importance available"}
        except Exception as e:
            print(f"❌ Error getting feature importance: {e}")
            return {"error": str(e)}

    def get_model_info(self):
        """Get comprehensive model information"""
        return {
            "predictors_count": len(self.predictors),
            "feature_names": self.feature_names,
            "training_date": self.training_date.isoformat()
            if self.training_date
            else "Unknown",
            "model_type": "XGBoost Classifier",
            "status": "trained" if self.model else "not_trained",
        }

    def run_full_pipeline(self):
        """Run the complete prediction pipeline"""
        print("🚀 Starting Bitcoin prediction pipeline...")

        # Step 1: Load data
        btc_data = self.load_data()
        if btc_data.empty:
            print("❌ Pipeline failed: No Bitcoin data loaded")
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {"up_probability": 50.0, "down_probability": 50.0},
                "error": "No Bitcoin data available",
            }

        # Step 2: Merge with sentiment
        merged_data = self.merge_with_sentiment(btc_data)
        if merged_data.empty:
            print("❌ Pipeline failed: No merged data")
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {"up_probability": 50.0, "down_probability": 50.0},
                "error": "Failed to merge data",
            }

        # Step 3: Create features
        enhanced_data = self.create_features(merged_data)
        if enhanced_data.empty:
            print("❌ Pipeline failed: No enhanced data")
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {"up_probability": 50.0, "down_probability": 50.0},
                "error": "Failed to create features",
            }

        # Step 4: Train model
        model = self.train_model(enhanced_data)
        if model is None:
            print("❌ Pipeline failed: Model training failed")
            return {
                "prediction": "UP",
                "confidence": 50.0,
                "current_price": 0.0,
                "prediction_proba": {"up_probability": 50.0, "down_probability": 50.0},
                "error": "Model training failed",
            }

        # Step 5: Make prediction
        prediction = self.predict_next_day(enhanced_data)

        print("✅ Prediction pipeline complete!")
        return prediction

    # new stratigy to improve confidence:
    def backtest(self, data, start=1095, step=150):
        """Walk-forward backtesting for realistic performance evaluation"""
        print("🎯 Running walk-forward backtesting...")

        all_predictions = []

        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i : (i + step)].copy()

            # Train temporary model on training period
            temp_model = XGBClassifier(
                random_state=42, learning_rate=0.1, n_estimators=100
            )
            temp_model.fit(train[self.predictors], train["target"])

            # Predict on test period
            preds = temp_model.predict(test[self.predictors])
            preds_series = pd.Series(preds, index=test.index, name="predictions")

            combined = pd.concat([test["target"], preds_series], axis=1)
            all_predictions.append(combined)

        return pd.concat(all_predictions)

    def compute_rolling_features(self, data):
        """Enhanced feature engineering from the profitable strategy"""
        print("🔄 Computing enhanced rolling features...")

        horizons = [2, 7, 60, 365]
        new_predictors = [
            "close",
            "volume",
            "open",
            "high",
            "low",
            "edit_count",
            "sentiment",
            "neg_sentiment",
        ]

        # Create target first (needed for trend features)
        data["tomorrow"] = data["close"].shift(-1)
        data["target"] = (data["tomorrow"] > data["close"]).astype(int)
        data = data.dropna(subset=["target"])

        for horizon in horizons:
            # Rolling averages
            rolling_averages = data.rolling(horizon, min_periods=1).mean()

            # Close ratio (same as before but more efficient)
            ratio_column = f"close_ratio_{horizon}"
            data[ratio_column] = data["close"] / rolling_averages["close"]

            # Edit count rolling
            edit_column = f"edit_{horizon}"
            data[edit_column] = rolling_averages["edit_count"]

            # ENHANCED: Trend based on actual target (this is the key
            # improvement!)
            rolling_target = data.rolling(horizon, closed="left", min_periods=1).mean()
            trend_column = f"trend_{horizon}"
            data[trend_column] = rolling_target["target"]

            new_predictors.extend([ratio_column, trend_column, edit_column])

        self.predictors = new_predictors
        self.feature_names = new_predictors

        print(f"✅ Enhanced feature engineering: {len(new_predictors)} features")
        return data
