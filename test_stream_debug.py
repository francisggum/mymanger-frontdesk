import requests
import json

def test_streaming():
    try:
        response = requests.post('http://localhost:8000/chat-stream', 
                               json={'query': 'test'}, 
                               stream=True, 
                               timeout=10)
        print('Status:', response.status_code)
        print('Headers:', dict(response.headers))
        
        if response.status_code == 200:
            print('Streaming response:')
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    print('Line:', line)
        else:
            print('Response:', response.text)
            
    except Exception as e:
        print('Error:', e)

if __name__ == "__main__":
    test_streaming()