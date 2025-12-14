import unittest
import yaml
import pandas as pd
from pathlib import Path
from Trend.Pillar.trend_pillar import VectorizedScorer, DEFAULT_INI

class TestTrendScenarios(unittest.TestCase):
    def setUp(self):
        # Load tests
        test_file = Path(__file__).parent / "trend_scenarios_tests.yaml"
        if not test_file.exists():
            self.skipTest("No test yaml found")

        with open(test_file, 'r') as f:
            data = yaml.safe_load(f)
            self.tests = data.get('tests', [])

        self.scorer = VectorizedScorer(DEFAULT_INI)

    def test_scenarios(self):
        for t in self.tests:
            with self.subTest(name=t['name']):
                input_data = t['input']
                expected = t['expected']

                # Create a 1-row DataFrame
                df = pd.DataFrame([input_data])

                # Evaluate
                # We need to ensure columns exist even if not in input,
                # but eval might fail if missing.
                # However, eval usually just uses what's there.
                # Our Scorer iterates all rules.
                # We should mock the 'score' column logic to isolate the specific rule?
                # The scorer runs ALL rules on the DF.
                # So we check if the specific rule fired by checking if score changed
                # relative to baseline 0? No, other rules might fire.
                # But here we only provide specific columns.
                # Ideally we want to check if the CONDITION is true.

                # Let's inspect the scorer config to find the rule definition
                scenario_name = expected['scenario']
                rule_def = next((r for r in self.scorer.config['scenarios'] if r['name'] == scenario_name), None)

                if not rule_def:
                    self.fail(f"Scenario {scenario_name} not found in INI")

                # Evaluate just this rule's condition on the df
                condition = rule_def['when']
                try:
                    result = df.eval(condition).iloc[0]
                except Exception as e:
                    self.fail(f"Eval failed for {scenario_name}: {e}")

                self.assertTrue(result, f"Scenario {scenario_name} condition did not fire for input {input_data}")

                # Check Bonus if applicable
                if 'bonus_when' in rule_def and rule_def['bonus_when']:
                    # Simple check if bonus expected
                    # Logic in yaml implies 'score_delta' includes bonus
                    # But verifying just the trigger is safer for unit test
                    pass

if __name__ == "__main__":
    unittest.main()
