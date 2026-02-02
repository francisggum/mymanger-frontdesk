import requests
import json
import time

def test_backend_streaming():
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
            results = []
            
            for i, line in enumerate(stream_response.iter_lines(decode_unicode=False)):
                if line:
                    try:
                        line_text = line.decode('utf-8', errors='ignore')
                        if line_text.startswith('data: '):
                            data = json.loads(line_text[6:])
                            status = data.get('status', 'unknown')
                            progress = data.get('progress', 0)
                            message = data.get('message', '')
                            
                            # 결과 저장
                            result_line = f'{i+1:2d}. [{status}] {progress:5.1f}% - {message[:100]}'
                            results.append(result_line)
                            print(result_line)
                            
                            if status == 'complete':
                                response_length = len(data.get('response', ''))
                                print(f'응답 길이: {response_length}')
                                break
                            elif status == 'error':
                                print(f'오류: {data.get("message", "")}')
                                break
                                
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f'처리 오류: {e}')
                        continue
                        
                    if i > 50:  # 무한 루프 방지
                        break
                        
            # 결과를 파일에 저장
            with open('streaming_test_result.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(results))
            print(f'\n결과가 streaming_test_result.txt 파일에 저장되었습니다.')
            
        else:
            print('스트리밍 실패:', stream_response.status_code, stream_response.text)
            
    except Exception as e:
        print('테스트 실패:', e)

if __name__ == "__main__":
    test_backend_streaming()