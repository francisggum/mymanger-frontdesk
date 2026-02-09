"""
LLM Tool Functions for Insurance Data Search
보험 데이터 검색을 위한 LLM 도구 함수들
"""

import logging
from typing import Dict, Any, List, Optional
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)


def search_by_company_name(llm_data: Dict[str, Any], company_name: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    회사명으로 검색 (부분 일치)
    
    Args:
        llm_data: 보험 데이터
        company_name: 검색할 회사명
        limit: 최대 결과 수
        
    Returns:
        검색된 회사 정보 리스트
    """
    results = []
    search_term = company_name.lower()
    
    for company_key, coverages in llm_data.items():
        # company_key 형식: "회사명_회사코드_상품명"
        parts = company_key.split("_")
        if len(parts) >= 1:
            key_company_name = parts[0].lower()
            
            # 부분 일치 검사
            if search_term in key_company_name or fuzz.partial_ratio(search_term, key_company_name) > 70:
                results.append({
                    "company_key": company_key,
                    "company_name": parts[0] if len(parts) > 0 else "",
                    "company_code": parts[1] if len(parts) > 1 else "",
                    "product_name": parts[2] if len(parts) > 2 else "",
                    "coverages": coverages,
                    "total_coverages": len(coverages),
                    "total_premium": sum(c.get("sum_premium", 0) for c in coverages)
                })
    
    # 유사도 순으로 정렬
    results.sort(key=lambda x: fuzz.partial_ratio(search_term, x["company_name"].lower()), reverse=True)
    
    return results[:limit]


def search_by_coverage_name(llm_data: Dict[str, Any], coverage_name: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    보장 항목명으로 검색 (부분 일치)
    
    Args:
        llm_data: 보험 데이터
        coverage_name: 검색할 보장 항목명
        limit: 최대 결과 수
        
    Returns:
        검색된 보장 항목 정보 리스트 (회사 정보 포함)
    """
    results = []
    search_term = coverage_name.lower()
    
    for company_key, coverages in llm_data.items():
        # company_key 형식: "회사명_회사코드_상품명"
        parts = company_key.split("_")
        company_name = parts[0] if len(parts) > 0 else ""
        company_code = parts[1] if len(parts) > 1 else ""
        product_name = parts[2] if len(parts) > 2 else ""
        
        for coverage in coverages:
            cov_name = coverage.get("coverage_name", "").lower()
            cov_code = coverage.get("coverage_code", "")
            
            # 부분 일치 검사
            if search_term in cov_name or fuzz.partial_ratio(search_term, cov_name) > 70:
                results.append({
                    "company_key": company_key,
                    "company_name": company_name,
                    "company_code": company_code,
                    "product_name": product_name,
                    "coverage_name": coverage.get("coverage_name"),
                    "coverage_code": cov_code,
                    "guide_contract_amount_max": coverage.get("guide_contract_amount_max"),
                    "sum_premium": coverage.get("sum_premium"),
                    "insur_item_list": coverage.get("insur_item_list", []),
                    "match_score": fuzz.partial_ratio(search_term, cov_name)
                })
    
    # 유사도 순으로 정렬
    results.sort(key=lambda x: x["match_score"], reverse=True)
    
    return results[:limit]


def get_company_summary(llm_data: Dict[str, Any], company_key: str) -> Optional[Dict[str, Any]]:
    """
    특정 회사의 요약 정보 조회
    
    Args:
        llm_data: 보험 데이터
        company_key: 회사 키 ("회사명_회사코드_상품명" 형식)
        
    Returns:
        회사 요약 정보
    """
    if company_key not in llm_data:
        return None
    
    coverages = llm_data[company_key]
    parts = company_key.split("_")
    
    # 총 보험료 계산
    total_premium = sum(c.get("sum_premium", 0) for c in coverages)
    
    # 보장 항목별 요약
    coverage_summary = []
    for c in coverages:
        coverage_summary.append({
            "coverage_name": c.get("coverage_name"),
            "coverage_code": c.get("coverage_code"),
            "sum_premium": c.get("sum_premium"),
            "guide_contract_amount_max": c.get("guide_contract_amount_max"),
            "item_count": len(c.get("insur_item_list", []))
        })
    
    return {
        "company_key": company_key,
        "company_name": parts[0] if len(parts) > 0 else "",
        "company_code": parts[1] if len(parts) > 1 else "",
        "product_name": parts[2] if len(parts) > 2 else "",
        "total_coverages": len(coverages),
        "total_premium": total_premium,
        "average_premium": total_premium / len(coverages) if coverages else 0,
        "coverages": coverage_summary
    }


def get_coverage_details(llm_data: Dict[str, Any], company_key: str, coverage_code: str) -> Optional[Dict[str, Any]]:
    """
    특정 회사의 특정 보장 항목 상세 정보 조회
    
    Args:
        llm_data: 보험 데이터
        company_key: 회사 키
        coverage_code: 보장 코드
        
    Returns:
        보장 항목 상세 정보
    """
    if company_key not in llm_data:
        return None
    
    coverages = llm_data[company_key]
    parts = company_key.split("_")
    
    for coverage in coverages:
        if coverage.get("coverage_code") == coverage_code:
            return {
                "company_key": company_key,
                "company_name": parts[0] if len(parts) > 0 else "",
                "company_code": parts[1] if len(parts) > 1 else "",
                "product_name": parts[2] if len(parts) > 2 else "",
                "coverage_name": coverage.get("coverage_name"),
                "coverage_code": coverage.get("coverage_code"),
                "guide_contract_amount_max": coverage.get("guide_contract_amount_max"),
                "sum_premium": coverage.get("sum_premium"),
                "insur_item_list": coverage.get("insur_item_list", [])
            }
    
    return None


def compare_companies_by_coverage(llm_data: Dict[str, Any], coverage_name: str) -> List[Dict[str, Any]]:
    """
    특정 보장 항목으로 모든 회사 비교
    
    Args:
        llm_data: 보험 데이터
        coverage_name: 비교할 보장 항목명
        
    Returns:
        회사별 해당 보장 항목 비교 결과
    """
    results = []
    search_term = coverage_name.lower()
    
    for company_key, coverages in llm_data.items():
        parts = company_key.split("_")
        company_name = parts[0] if len(parts) > 0 else ""
        company_code = parts[1] if len(parts) > 1 else ""
        product_name = parts[2] if len(parts) > 2 else ""
        
        for coverage in coverages:
            cov_name = coverage.get("coverage_name", "").lower()
            
            # 부분 일치 검사
            if search_term in cov_name or fuzz.partial_ratio(search_term, cov_name) > 80:
                results.append({
                    "company_key": company_key,
                    "company_name": company_name,
                    "company_code": company_code,
                    "product_name": product_name,
                    "coverage_name": coverage.get("coverage_name"),
                    "coverage_code": coverage.get("coverage_code"),
                    "sum_premium": coverage.get("sum_premium"),
                    "guide_contract_amount_max": coverage.get("guide_contract_amount_max"),
                    "insur_item_list": coverage.get("insur_item_list", [])
                })
    
    # 보험료 순으로 정렬 (저렴한 순)
    results.sort(key=lambda x: x.get("sum_premium", float('inf')))
    
    return results


def get_cheapest_company(llm_data: Dict[str, Any], coverage_name: str) -> Optional[Dict[str, Any]]:
    """
    특정 보장 항목에서 가장 저렴한 회사 조회
    
    Args:
        llm_data: 보험 데이터
        coverage_name: 보장 항목명
        
    Returns:
        가장 저렴한 회사 정보
    """
    comparisons = compare_companies_by_coverage(llm_data, coverage_name)
    
    if not comparisons:
        return None
    
    return comparisons[0]


def list_all_companies(llm_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    모든 회사 목록 조회
    
    Args:
        llm_data: 보험 데이터
        
    Returns:
        회사 목록
    """
    results = []
    
    for company_key, coverages in llm_data.items():
        parts = company_key.split("_")
        total_premium = sum(c.get("sum_premium", 0) for c in coverages)
        
        results.append({
            "company_key": company_key,
            "company_name": parts[0] if len(parts) > 0 else "",
            "company_code": parts[1] if len(parts) > 1 else "",
            "product_name": parts[2] if len(parts) > 2 else "",
            "total_coverages": len(coverages),
            "total_premium": total_premium
        })
    
    # 회사명 순으로 정렬
    results.sort(key=lambda x: x["company_name"])
    
    return results


# Tool schemas for OpenAI function calling
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_by_company_name",
            "description": "회사명으로 보험사를 검색합니다. 부분 일치 지원.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "검색할 회사명 (예: 'DB', '삼성', '현대')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "최대 결과 수 (기본값: 10)",
                        "default": 10
                    }
                },
                "required": ["company_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_coverage_name",
            "description": "보장 항목명으로 검색합니다. 예: '암진단', '상해사망', '질병후유장해'",
            "parameters": {
                "type": "object",
                "properties": {
                    "coverage_name": {
                        "type": "string",
                        "description": "검색할 보장 항목명"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "최대 결과 수 (기본값: 20)",
                        "default": 20
                    }
                },
                "required": ["coverage_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_summary",
            "description": "특정 회사의 전체 보장 항목과 총 보험료 요약 정보를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_key": {
                        "type": "string",
                        "description": "회사 키 (형식: '회사명_회사코드_상품명')"
                    }
                },
                "required": ["company_key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_coverage_details",
            "description": "특정 회사의 특정 보장 항목 상세 정보를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company_key": {
                        "type": "string",
                        "description": "회사 키"
                    },
                    "coverage_code": {
                        "type": "string",
                        "description": "보장 코드 (예: 'b001', 'a012')"
                    }
                },
                "required": ["company_key", "coverage_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_companies_by_coverage",
            "description": "특정 보장 항목으로 모든 회사를 비교합니다. 보험료가 저렴한 순으로 정렬됩니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coverage_name": {
                        "type": "string",
                        "description": "비교할 보장 항목명 (예: '통합암진단비', '상해사망')"
                    }
                },
                "required": ["coverage_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cheapest_company",
            "description": "특정 보장 항목에서 가장 저렴한 회사를 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "coverage_name": {
                        "type": "string",
                        "description": "보장 항목명"
                    }
                },
                "required": ["coverage_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_companies",
            "description": "모든 보험사 목록을 조회합니다.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


# Tool schemas for Gemini function calling
GEMINI_TOOLS = [
    {
        "name": "search_by_company_name",
        "description": "회사명으로 보험사를 검색합니다. 부분 일치 지원.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "검색할 회사명 (예: 'DB', '삼성', '현대')"
                },
                "limit": {
                    "type": "integer",
                    "description": "최대 결과 수 (기본값: 10)"
                }
            },
            "required": ["company_name"]
        }
    },
    {
        "name": "search_by_coverage_name",
        "description": "보장 항목명으로 검색합니다. 예: '암진단', '상해사망', '질병후유장해'",
        "parameters": {
            "type": "object",
            "properties": {
                "coverage_name": {
                    "type": "string",
                    "description": "검색할 보장 항목명"
                },
                "limit": {
                    "type": "integer",
                    "description": "최대 결과 수 (기본값: 20)"
                }
            },
            "required": ["coverage_name"]
        }
    },
    {
        "name": "get_company_summary",
        "description": "특정 회사의 전체 보장 항목과 총 보험료 요약 정보를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_key": {
                    "type": "string",
                    "description": "회사 키 (형식: '회사명_회사코드_상품명')"
                }
            },
            "required": ["company_key"]
        }
    },
    {
        "name": "get_coverage_details",
        "description": "특정 회사의 특정 보장 항목 상세 정보를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "company_key": {
                    "type": "string",
                    "description": "회사 키"
                },
                "coverage_code": {
                    "type": "string",
                    "description": "보장 코드 (예: 'b001', 'a012')"
                }
            },
            "required": ["company_key", "coverage_code"]
        }
    },
    {
        "name": "compare_companies_by_coverage",
        "description": "특정 보장 항목으로 모든 회사를 비교합니다. 보험료가 저렴한 순으로 정렬됩니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "coverage_name": {
                    "type": "string",
                    "description": "비교할 보장 항목명 (예: '통합암진단비', '상해사망')"
                }
            },
            "required": ["coverage_name"]
        }
    },
    {
        "name": "get_cheapest_company",
        "description": "특정 보장 항목에서 가장 저렴한 회사를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "coverage_name": {
                    "type": "string",
                    "description": "보장 항목명"
                }
            },
            "required": ["coverage_name"]
        }
    },
    {
        "name": "list_all_companies",
        "description": "모든 보험사 목록을 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]


# Function mapping for execution
TOOL_FUNCTIONS = {
    "search_by_company_name": search_by_company_name,
    "search_by_coverage_name": search_by_coverage_name,
    "get_company_summary": get_company_summary,
    "get_coverage_details": get_coverage_details,
    "compare_companies_by_coverage": compare_companies_by_coverage,
    "get_cheapest_company": get_cheapest_company,
    "list_all_companies": list_all_companies
}


def execute_tool(llm_data: Dict[str, Any], tool_name: str, parameters: Dict[str, Any]) -> Any:
    """
    도구 함수 실행
    
    Args:
        llm_data: 보험 데이터
        tool_name: 실행할 도구 이름
        parameters: 도구 파라미터
        
    Returns:
        도구 실행 결과
    """
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {tool_name}"}
    
    try:
        # 첫 번째 파라미터는 항상 llm_data
        func = TOOL_FUNCTIONS[tool_name]
        return func(llm_data, **parameters)
    except Exception as e:
        logger.error(f"Tool execution error ({tool_name}): {e}")
        return {"error": f"Tool execution failed: {str(e)}"}
