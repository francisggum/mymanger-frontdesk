import requests
import json
import time

def test_backend_connection():
    print('백엔드 서버 연결 테스트...')
    time.sleep(2)  # 서버 시작 대기
    
    try:
        response = requests.get('http://localhost:8000/', timeout=5)
        print('서버 상태:', response.status_code)
        print('응답:', response.json())
        
        # 스트리밍 테스트
        print('\n스트리밍 테스트 시작...')
        stream_response = requests.post('http://localhost:8000/chat-stream', 
                                       json={'query': '가장 저렴한 보험 추천해줘'}, 
                                       stream=True, 
                                       timeout=30)
        
        if stream_response.status_code == 200:
            print('스트리밍 응답 수신 중...')
            for i, line in enumerate(stream_response.iter_lines(decode_unicode=True)):
                if line and line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        status = data.get('status', 'unknown')
                        progress = data.get('progress', 0)
                        message = data.get('message', '')
                        
                        print(f'{i+1:2d}. [{status}] {progress:5.1f}% - {message[:50]}')
                        
                        if status == 'complete':
                            print('응답 길이:', len(data.get('response', '')))
                            break
                        elif status == 'error':
                            print('오류:', data.get('message', ''))
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
                    if i > 20:  # 무한 루프 방지
                        break
        else:
            print('스트리밍 실패:', stream_response.status_code, stream_response.text)
            
    except Exception as e:
        print('테스트 실패:', e)

if __name__ == "__main__":
    test_backend_connection()