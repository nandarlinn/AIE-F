import pandas as pd
import json
import os
import argparse
from sklearn.metrics import classification_report

# ==========================================
# 1. RDR Node Structure
# ==========================================
class RDRNode:
    def __init__(self, condition=None, conclusion=None):
        self.condition = condition 
        self.conclusion = str(conclusion) if conclusion is not None else None
        self.if_true = None   # Refinement (Exception)
        self.if_false = None  # Alternative

    def evaluate(self, row):
        if not self.condition or not self.condition['col']:
            return True
        col, op, val = self.condition['col'], self.condition['op'], self.condition['val']
        row_val = row[col]
        try:
            if op == '==': return str(row_val) == str(val)
            if op == '<':  return float(row_val) < float(val)
            if op == '>':  return float(row_val) > float(val)
            if op == '>=': return float(row_val) >= float(val)
            if op == '<=': return float(row_val) <= float(val)
        except:
            return str(row_val) == str(val)
        return False

    def to_dict(self):
        return {
            "condition": self.condition,
            "conclusion": self.conclusion,
            "if_true": self.if_true.to_dict() if self.if_true else None,
            "if_false": self.if_false.to_dict() if self.if_false else None
        }

    @staticmethod
    def from_dict(data):
        if not data: return None
        node = RDRNode(data['condition'], data['conclusion'])
        node.if_true = RDRNode.from_dict(data['if_true'])
        node.if_false = RDRNode.from_dict(data['if_false'])
        return node

# ==========================================
# 2. SCRDR Engine
# ==========================================
class SCRDR_Engine:
    def __init__(self, target="Result", default_conclusion="Unknown"):
        self.target = target
        self.root = RDRNode(conclusion=default_conclusion)

    def classify(self, row):
        curr = self.root
        last_match = self.root
        while curr:
            if curr.evaluate(row):
                last_match = curr
                curr = curr.if_true
            else:
                curr = curr.if_false
        return last_match

    def add_rule(self, row, last_node):
        print(f"\n[KNOWLEDGE ACQUISITION]")
        print(f"System predicted: '{last_node.conclusion}' but Actual is: '{row[self.target]}'")
        
        while True:
            col = input(f"Enter feature name (e.g., E_Score, S_Score) or 'exit': ").strip()
            if col.lower() == 'exit': return False
            if col in row.index:
                break
            print(f"Error: '{col}', E_Score, S_Score, T_Score, J_Score (or) Q1-Q20")
            
        op = input("Enter operator (==, <, >, <=, >=): ").strip()
        val = input("Enter threshold value (e.g., 15): ").strip()
        
        new_node = RDRNode(condition={'col': col, 'op': op, 'val': val}, conclusion=row[self.target])
        
        if last_node.evaluate(row):
            # Refinement Rule
            if last_node.if_true is None:
                last_node.if_true = new_node
            else:
                curr = last_node.if_true
                while curr.if_false: curr = curr.if_false
                curr.if_false = new_node
        else:
            # Alternative Rule
            curr = last_node
            while curr.if_false: curr = curr.if_false
            curr.if_false = new_node
        return True

# ==========================================
# 3. Interactive Quiz Mode (UI Simulation)
# ==========================================
def run_quiz(engine):
    print("\n" + "="*50)
    print("Welcome to MBTI Personality Test (RDR Expert System)")
    print("="*50)
    print("Please answer with a score from 1 to 5.")
    print("(1 = Strongly Disagree, 5 = Strongly Agree)\n")
    
    # Reading questions.csv
    try:
        q_df = pd.read_csv("questions.csv")
    except Exception as e:
        print("Error: 'questions.csv' file not found.")
        return

    user_answers = {}
    
    # Looping through each question
    for idx, row in q_df.iterrows():
        q_id = row['Question_ID']
        q_text = row['Question_Text']
        
        while True:
            try:
                # Display full question text instead of Q1, Q2
                ans = int(input(f"{q_id}. {q_text} \nScore (1-5): "))
                if 1 <= ans <= 5:
                    user_answers[q_id] = ans
                    print("-" * 30)
                    break
                else:
                    print("Please enter a number between 1 and 5.\n")
            except ValueError:
                print("Invalid input. Please enter a number.\n")
                
    # Calculating Score dimension
    user_answers['E_Score'] = sum(user_answers[f'Q{i}'] for i in range(1, 6))
    user_answers['S_Score'] = sum(user_answers[f'Q{i}'] for i in range(6, 11))
    user_answers['T_Score'] = sum(user_answers[f'Q{i}'] for i in range(11, 16))
    user_answers['J_Score'] = sum(user_answers[f'Q{i}'] for i in range(16, 21))

    result_node = engine.classify(user_answers)
    print("\n" + "="*50)
    print(f"Based on the SCRDR rules, your MBTI Personality Type is: -> {result_node.conclusion} <-")
    print("="*50 + "\n")

# ==========================================
# 4. Main Function
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="MBTI SCRDR System")
    parser.add_argument("--mode", choices=['build', 'test', 'quiz'], default='quiz', 
                        help="build: Train rules | test: Evaluate accuracy | quiz: Take the test interactively")
    parser.add_argument("--input", default="mbti_100_rows.csv", help="Dataset CSV file")
    parser.add_argument("--tree", default="mbti_rdr_model.json", help="Saved RDR rules file")
    args = parser.parse_args()

    engine = SCRDR_Engine(target="Result")
    
    # Load existing rules if available
    if os.path.exists(args.tree):
        with open(args.tree, 'r') as f:
            engine.root = RDRNode.from_dict(json.load(f))
        print(f"[INFO] Loaded existing RDR model: {args.tree}")

    if args.mode == 'quiz':
        if not os.path.exists(args.tree):
            print("Warning: No rule model found! Please run '--mode build' first to create rules.")
        run_quiz(engine)
        return

    # Build or Test Mode needs CSV Data
    try:
        df = pd.read_csv(args.input)
        
        # Calculate Score dimension
        df['E_Score'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].sum(axis=1)
        df['S_Score'] = df[['Q6', 'Q7', 'Q8', 'Q9', 'Q10']].sum(axis=1)
        df['T_Score'] = df[['Q11', 'Q12', 'Q13', 'Q14', 'Q15']].sum(axis=1)
        df['J_Score'] = df[['Q16', 'Q17', 'Q18', 'Q19', 'Q20']].sum(axis=1)

    except Exception as e:
        print(f"Error loading CSV '{args.input}': {e}")
        print("Please make sure you have generated the 'mbti_100_rows.csv' file.")
        return

    y_true, y_pred = [], []
    stopped_early = False

    print(f"\n--- Running in {args.mode.upper()} mode ---")
    for idx, row in df.iterrows():
        pred_node = engine.classify(row)
        actual = str(row["Result"])
        predicted = str(pred_node.conclusion)
        
        if args.mode == 'build' and predicted != actual:
            print(f"\n[Row {idx+1}/{len(df)}] ERROR Found!")
            
            # Show Score
            print("\n[ Score (Score > 15 means the first letter) ]")
            print(f"E_Score: {row.get('E_Score', 0):>2}/25  -> (Extravert [E] vs Introvert [I])")
            print(f"S_Score: {row.get('S_Score', 0):>2}/25  -> (Sensing [S] vs Intuition [N])")
            print(f"T_Score: {row.get('T_Score', 0):>2}/25  -> (Thinking [T] vs Feeling [F])")
            print(f"J_Score: {row.get('J_Score', 0):>2}/25  -> (Judging [J] vs Perceiving [P])")
            print("-" * 50)
            # -------------------------------------------------------------
            
            success = engine.add_rule(row, pred_node)
            if not success:
                print("\nExiting and saving model...")
                stopped_early = True
                break
                
            # Save progress
            with open(args.tree, 'w') as f:
                json.dump(engine.root.to_dict(), f, indent=2)
            
            predicted = str(engine.classify(row).conclusion)

        y_true.append(actual)
        y_pred.append(predicted)

    print("\n--- PERFORMANCE SUMMARY ---")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    main()