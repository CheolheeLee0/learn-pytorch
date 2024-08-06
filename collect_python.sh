find . -not -path "./db/migrations/*" \( -name "*.go" -o -name "*.sql" \) -type f -print0 | while IFS= read -r -d '' file; 
do 
    echo "File: $file";
    echo "Content:"; 
    sed 's|--.*||g; s|//.*||g; /\/\*.*\*\//d; /\/\*/,/\*\//d' "$file"
    echo
done > go_files_contents.txt