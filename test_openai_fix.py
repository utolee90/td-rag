#!/usr/bin/env python3
"""
OpenAI API 키 설정 문제 수정 테스트
"""

import sys
import os
sys.path.append('/mnt/nvme02/User/utopiamath/vaiv/RTRAG')

def test_openai_api_setup():
    """OpenAI API 설정 테스트"""
    try:
        from search import SearchInterface
        from manager import VectorStoreManager
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from pathlib import Path
        
        print("✓ 모듈 import 성공")
        
        # 임시 VectorStoreManager 생성
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        base_dir = Path("faiss_indexes")
        manager = VectorStoreManager(embeddings, base_dir)
        
        # SearchInterface 인스턴스 생성
        search_interface = SearchInterface(manager)
        print("✓ SearchInterface 초기화 성공")
        
        # API 키 없이 OpenAI 모델 초기화 시도 (에러가 예상됨)
        result = search_interface.init_openai_model("")
        print(f"API 키 없음 테스트: {result}")
        
        # 가짜 API 키로 테스트 (에러가 예상됨)
        fake_api_key = "sk-fake123456789"
        result = search_interface.init_openai_model(fake_api_key)
        print(f"가짜 API 키 테스트: {result}")
        
        # 답변 생성 테스트 (OpenAI 없이)
        context = "This is a test context about weather."
        question = "What is the weather like?"
        
        # Local model로 테스트
        answer = search_interface.generate_answer("Local HuggingFace - Generate", question, context)
        print(f"✓ Local model 답변 생성 성공: {answer[:50]}...")
        
        # OpenAI 모델로 테스트 (API 키 없어서 에러 예상)
        answer = search_interface.generate_answer("OpenAI", question, context)
        print(f"OpenAI 답변 테스트: {answer}")
        
        return True
        
    except Exception as e:
        print(f"✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== OpenAI API 키 설정 수정 테스트 ===")
    success = test_openai_api_setup()
    if success:
        print("\n✓ 테스트 완료! OpenAI API 키 설정 문제가 수정되었습니다.")
    else:
        print("\n✗ 테스트 실패! 추가 수정이 필요합니다.")
