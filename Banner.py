FILE_SEQUENCE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'
BANNER_LETTERS = {}

with open('banner_data.txt', 'r', encoding='utf-8') as banner_file:
    data = banner_file.readlines()

cleaned = []
for d in data:
    cleaned.append(d.replace('\n', ''))

cleaned.reverse()

letter_value = 0
this_letter = []
this_letter.clear()
while len(cleaned):
    this_line = cleaned.pop()
    if this_line == '                    ' and letter_value <= len(FILE_SEQUENCE):
        this_letter.append('                    ')
        BANNER_LETTERS[FILE_SEQUENCE[letter_value]] = [_ for _ in this_letter]
        letter_value += 1
        this_letter.clear()
        while len(cleaned):
            if cleaned[-1] == '                    ':
                cleaned.pop()
            else:
                break
    else:
        this_letter.append(this_line)

for letter in BANNER_LETTERS:
    this_letter = BANNER_LETTERS[letter]

    max_length = max([len(_) for _ in this_letter])

    # Check if all lines have spaces at the end
    while max_length > 0:
        found_character = False
        for line in this_letter:
            if line[max_length - 1] != ' ':
                found_character = True
        if not found_character:
            max_length -= 1
        else:
            break

    new_letter = [x[:max_length + 1] for x in this_letter]
    BANNER_LETTERS[letter] = [_ for _ in new_letter]

BANNER_LETTERS[' '] = ['      '] * 9
message = 'abc def 123'
message = message.upper()
print(message)

banner_message = []
for char in message:
    letter_to_add = BANNER_LETTERS[char]
    if len(banner_message) == 0:
        banner_message = [_ for _ in letter_to_add]
    else:
        for i, row in enumerate(BANNER_LETTERS[char]):
            banner_message[i] += row
    # break

for m in banner_message:
    print(m)
