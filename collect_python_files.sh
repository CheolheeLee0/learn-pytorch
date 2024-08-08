find . -not -path "./venv/*" \( -name "*.py" \) -type f -print0 | while IFS= read -r -d '' file; 
do 
    echo "File: $file";
    echo "Content:"; 
    sed 's|#.*||g' "$file"  # Python 주석 제거
    echo
done > python_files_contents.txt