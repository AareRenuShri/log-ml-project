import random
import csv

templates = {
    "timeout": [
        "Connection to server {id} timed out after 30s",
        "Request {id} exceeded timeout limit of 5000ms",
    ],
    "auth_failure": [
        "Authentication failed for user_{id}: invalid credentials",
        "Login attempt {id} rejected: token expired",
    ],
    "null_pointer": [
        "NullPointerException at line {id} in module handler",
        "Null reference encountered while processing object_{id}",
    ],
    "normal": [
        "Request {id} processed successfully in 120ms",
        "User_{id} logged in successfully",
    ],
}

rows = []

for label, patterns in templates.items(): 
    for i in range(300):
        chosen_pattern = random.choice(patterns)
        random_id = random.randint(1000, 9999)
        log_line = chosen_pattern.format(id= random_id)
        rows.append((log_line, label))

with open("logs_labeled.csv", "w", newline = "") as f:
    writer = csv.writer(f)
    writer.writerow(["log_text", "label"])
    writer.writerows(rows)
    
print(f"Wrote {len(rows)} rows to logs_labeled.csv")