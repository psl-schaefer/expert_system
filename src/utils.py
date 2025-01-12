
def pretty_print_text(text, max_chars_per_line=80):
    lines = text.split('\n')
    for line in lines:
        words = line.split()
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line += word + " "
            else:
                print(current_line.strip())
                current_line = word + " "

        if current_line:
            print(current_line.strip())
        print()
        