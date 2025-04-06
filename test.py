import unittest
import pandas as pd

# Import the real functions from your modules
from data_preprocessing import load_and_preprocess_data
from recommendation_engine import recommend_assessments, evaluate_recommendations

class TestRecommendationMetrics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load data once for all tests
        try:
            cls.df, cls.model, cls.index = load_and_preprocess_data()
            print("Data, model, and index loaded successfully!")
        except Exception as e:
            raise unittest.SkipTest(f"Skipping tests due to load failure: {str(e)}")

    def setUp(self):
        # Sample query and relevant assessments from your example
        self.single_query = (
            "We are seeking a Junior Java Developer to join our development team. "
            "The ideal candidate will be responsible for writing clean, efficient Java code, "
            "collaborating with business analysts and product managers to design software solutions, "
            "and participating in code reviews. Key tasks include developing and maintaining "
            "web applications using Java frameworks (e.g., Spring), troubleshooting and debugging "
            "applications, and working effectively in a team environment to meet project deadlines."
        )
        self.queries = [
            "Hiring for Java developers with collaboration skills",
            "Administrative professional test"
        ]
        self.relevant = [
            ["Java Web Services (New)", "Java 8 (New)"],
            ["Administrative Professional - Short Form"]
        ]

    def test_single_query_recommendations(self):
        # Test single query recommendation with real prediction
        recs = recommend_assessments(
            self.single_query, self.df, self.model, self.index,
            max_duration=40, preferred_test_types=['K', 'C']
        )
        
        print("\nSingle Query Recommendations:")
        print("Query:", self.single_query)
        print("Recommendations:", recs)
        
        self.assertTrue(isinstance(recs, list), "Recommendations should be a list")
        self.assertTrue(len(recs) > 0, "Should return at least one recommendation")
        self.assertTrue(all(isinstance(r, dict) for r in recs), "Each recommendation should be a dict")
        self.assertTrue(all('Assessment Name' in r for r in recs), "Each recommendation should have an Assessment Name")
        self.assertTrue(all(r['Duration'] <= 40 for r in recs if 'Duration' in r), "All durations should be <= 40")
        self.assertTrue(all(r['Test Type'] in ['K', 'C'] for r in recs if 'Test Type' in r), "Test types should match preferred types")

    def test_evaluate_recommendations(self):
        # Test evaluate_recommendations with real predictions
        recall, ap = evaluate_recommendations(self.queries, self.relevant, self.df, self.model, self.index)
        
        print("\nMultiple Queries Evaluation:")
        print("Queries:", self.queries)
        print("Relevant:", self.relevant)
        print(f"Mean Recall@3: {recall:.4f}")
        print(f"Mean AP@3: {ap:.4f}")
        
        self.assertGreaterEqual(recall, 0.0, "Mean Recall@3 should be >= 0")
        self.assertLessEqual(recall, 1.0, "Mean Recall@3 should be <= 1")
        self.assertGreaterEqual(ap, 0.0, "Mean AP@3 should be >= 0")
        self.assertLessEqual(ap, 1.0, "Mean AP@3 should be <= 1")

if __name__ == '__main__':
    unittest.main(verbosity=2)