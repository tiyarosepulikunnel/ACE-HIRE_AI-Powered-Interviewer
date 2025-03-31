import re

# Read the original file
with open('app.py', 'r') as f:
    content = f.read()

# Replace all query_params navigations with session_state
content = content.replace('st.query_params["page"] = ', 'st.session_state["current_page"] = ')

# Write the modified content back
with open('app.py', 'w') as f:
    f.write(content)

pages = ["Upload Resume", "Resume-Job Match", "Generate Questions", "Interview Session", "Final Report"]