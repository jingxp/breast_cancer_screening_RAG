
def read_data() -> str:
    """
        Reads data from a file and returns it as a string.
        
        Returns:
            str: The entire contents of the file as a single string.
        """
    with open('paper.md', 'r', encoding='utf-8') as file:
        return file.read()

def get_chunks(data: str) -> list[str]:
    """
    Splits the input data into chunks of a specified size.
    Args:
        data (str): The input data to be split into chunks.
        
    Returns:
        list: A list of strings, where each string is a chunk of the input data.
    """
    content: str = read_data()
    chunks: list[str] = content.split('\n\n')

    result: list[any] = []
    header = ""
    for chunk in chunks:
        if chunk.startswith('**'):
            header += f"{chunk}\n"
        else:
            result.append(f"{header}{chunk}")
            header = ""
    return result

if __name__ == "__main__":
    data = read_data()
    chunks = get_chunks(data)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")
        print("-" * 40)  # Separator for readability
    print(f"Total chunks: {len(chunks)}")
    print("Data read successfully.")
    print("Chunks generated successfully.")
