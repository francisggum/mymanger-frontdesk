try:
    from langchain.chains.question_answering import load_qa_chain
    print('New import successful')
except ImportError as e:
    try:
        from langchain.chains import load_qa_chain
        print('Old import successful')
    except ImportError as e2:
        print(f'Both imports failed: {e}, {e2}')