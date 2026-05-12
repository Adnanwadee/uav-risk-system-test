#!/usr/bin/env python3
"""
RAG System Test with Detailed Debug Output
===========================================
"""

import sys
import asyncio
import logging
from pathlib import Path

# إعداد logging لعرض كل شيء
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# إضافة المسار
sys.path.insert(0, '/workspaces/uav-risk-system-test/src')
sys.path.insert(0, '/workspaces/uav-risk-system-test/src/uav_risk/stage2')

from uav_risk.stage2.rag.rag_core import AsyncRAGCore


async def test():
    print("\n" + "=" * 80)
    print("🧪 RAG SYSTEM TEST - WITH FULL DEBUG")
    print("=" * 80)
    
    # تهيئة النظام
    print("\n📡 Initializing RAG Core (Debug Mode ON)...")
    rag = AsyncRAGCore()
    
    if not rag._db or not rag._llm:
        print("❌ System not ready")
        return
    
    # أسئلة الاختبار
    test_queries = [
        ("FAA Altitude", "What is the maximum altitude for drones under FAA Part 107?"),
        ("FAA Night Ops", "What are the night operations requirements?"),
        ("FAA Pilot Cert", "What are the pilot certification requirements?"),
        ("Comparison", "Compare FAA and EASA requirements for remote pilot certification")
    ]
    
    print("\n" + "=" * 80)
    print("🚀 STARTING TESTS")
    print("=" * 80)
    
    for name, query in test_queries:
        print(f"\n{'='*60}")
        print(f"📋 TEST: {name}")
        print(f"❓ Query: {query}")
        print("=" * 60)
        
        result = await rag.ask_legal_question(query, top_k=4, min_score=0.4)
        
        print(f"\n📊 RESULT:")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Citations: {len(result.citations)}")
        
        if result.debug_info:
            print(f"   Debug Info: {result.debug_info}")
        
        print(f"\n📝 ANSWER:")
        print("-" * 40)
        print(result.answer[:800])
        print("-" * 40)
        
        print("\n" + "~" * 60)
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test())