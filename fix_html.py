import re

with open('david_streamlit.py', 'r', encoding='utf-8') as f:
    code = f.read()

def replacer(match):
    prefix = match.group(1)
    content = match.group(2)
    suffix = match.group(3)
    
    lines = content.split('\n')
    new_lines = [line.lstrip() for line in lines]
    
    new_content = '\n'.join(new_lines)
    return f'{prefix}{new_content}{suffix}'

# Match st.markdown(f""" ... """, unsafe_allow_html=True) AND st.markdown(f''' ... ''', unsafe_allow_html=True)
new_code = re.sub(r'(st\.markdown\(\s*[fF]?(?:"""|\'\'\'))(.*?)((?:"""|\'\'\').*?unsafe_allow_html=True\s*\))', replacer, code, flags=re.DOTALL)

with open('david_streamlit.py', 'w', encoding='utf-8') as f:
    f.write(new_code)
print("done")
