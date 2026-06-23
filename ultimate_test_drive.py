import polars as pl
import logging
import narwhals as nw

# Import your entire Phase 1-6 Arsenal
from cleanframe.pipeline import DataCleaner
from cleanframe.rules import (
    KNNImputationRule,
    FuzzyUnificationRule,
    CrossColumnConsistencyRule,
    NearDuplicateDetector
)
from cleanframe.rules.cross_column import ConsistencyConstraint
from cleanframe.telemetry.sinks import StandardLoggingSink, LocalJsonLinesSink

# ---------------------------------------------------------
# 1. THE CHAOS MATRIX
# ---------------------------------------------------------
dirty_data = pl.DataFrame({
    "user_id": [1, 2, 3, 4, 5, 6, 7],
    "full_name": ["Jon Doe", "Jane Smith", "Jon Doe", "Alice Jones", "Bob", "Charlie", "Alice J."], 
    "age": [45, None, 45, 16, None, 32, 16], 
    "role": ["Manager", "manager", "MANAGER", "Manager", "Employee", "employee", "Manager"], 
    "email": ["jon@corp.com", "jane@corp.com", "jon@corp.com", "invalid-email", "bob@corp.com", "char@corp.com", "alice@corp.com"], 
    "days_since_active": [2, 15, 2, 0, 100, 45, 0], 
    "churn_risk_score": [0.1, 0.9, 0.1, 0.0, 0.99, 0.5, 0.0], 
    "is_churned": [False, True, False, False, True, False, False] 
})

# ---------------------------------------------------------
# 2. THE RUTHLESS LOGIC ENFORCER
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Notice the action="drop" flag. The engine will no longer just watch; it will purge.
underage_manager_rule = ConsistencyConstraint(
    name="Underage Manager",
    condition=(nw.col("age") < 18) & (nw.col("role").str.to_lowercase() == "manager"),
    error_msg="Violation: User is under 18 but has a Manager role.",
    action="drop"  
)

# ---------------------------------------------------------
# 3. INITIALIZING THE EXPLICIT ENGINE
# ---------------------------------------------------------
cleaner = DataCleaner(
    rules=[
        KNNImputationRule(k=2, max_rows=10000),             
        # DX Tweak: Protect the IDs and Emails from being clustered! Lowercase roles before matching.
        FuzzyUnificationRule(threshold=85.0, exclude_cols=["email", "full_name", "user_id"], pre_lowercase=True),               
        CrossColumnConsistencyRule([underage_manager_rule]),
        NearDuplicateDetector(num_perm=32, threshold=0.8)   
    ],
    sinks=[
        StandardLoggingSink(),
        LocalJsonLinesSink("ultimate_audit.jsonl")          
    ]
)

# ---------------------------------------------------------
# 4. EXECUTION
# ---------------------------------------------------------
print("\n🔥 --- INITIATING CLEANFRAME ENFORCER PIPELINE --- 🔥\n")

clean_data = cleaner.fit_transform(dirty_data, target_col="is_churned")

print("\n==================================================")
print(f"🛡️ FINAL CLEANED DATA SHAPE: {clean_data.shape}")
print("==================================================")
print(clean_data)