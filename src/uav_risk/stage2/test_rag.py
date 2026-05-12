#!/usr/bin/env python3
"""
RAG System Test with Detailed Debug Output - Enhanced
"""

import sys
import asyncio
import logging
from pathlib import Path

# إضافة المسار الصحيح
sys.path.insert(0, '/workspaces/uav-risk-system-test/src')
sys.path.insert(0, '/workspaces/uav-risk-system-test/src/uav_risk/stage2')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

# استيراد مباشر (بدون نقطة)
from rag.rag_core import AsyncRAGCore   # ✅ صحيح

async def test():
    print("\n" + "=" * 80)
    print("🧪 RAG SYSTEM TEST - ENHANCED VERSION")
    print("=" * 80)
    
    print("\n📡 Initializing RAG Core...")
    rag = AsyncRAGCore()
    
    if not rag._db:
        print("❌ FAISS index not loaded")
        return
    
    if not rag._llm:
        print("❌ Groq LLM not loaded")
        print("   Check GROQ_API_KEY is set")
        return
    
    print("✅ System ready!\n")
    
    # أسئلة الاختبار
    test_queries = [
        ("FAA Altitude", "What is the maximum altitude for drones under FAA Part 107? What are the exceptions?"),
        ("Pilot Certification", "What are the complete pilot certification requirements under FAA Part 107?"),
    ]
    
    results_summary = []
    
    for name, query in test_queries:
        print(f"\n{'='*60}")
        print(f"📋 TEST: {name}")
        print(f"❓ Query: {query}")
        print("=" * 60)
        
        result = await rag.ask_legal_question(query, top_k=8, min_score=0.3)
        
        print(f"\n📊 RESULTS:")
        print(f"   Strategy: {result.debug_info.get('strategy', 'unknown')}")
        print(f"   Confidence: {result.confidence_score:.3f}")
        print(f"   Citations: {len(result.citations)}")
        print(f"   FAA chunks: {result.debug_info.get('faa_chunks', 0)}")
        print(f"   EASA chunks: {result.debug_info.get('easa_chunks', 0)}")
        
        results_summary.append({
            "name": name,
            "confidence": result.confidence_score,
            "citations": len(result.citations),
        })
        
        print(f"\n📝 ANSWER PREVIEW:")
        print("-" * 40)
        print(result.answer[:500])
        print("-" * 40)
    
    # ملخص
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    if results_summary:
        avg_confidence = sum(r["confidence"] for r in results_summary) / len(results_summary)
        avg_citations = sum(r["citations"] for r in results_summary) / len(results_summary)
        
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Average Citations: {avg_citations:.1f}")
        
        if avg_confidence >= 0.7:
            print("✅ Target achieved: Confidence > 0.7")
        else:
            print(f"⚠️ Current: {avg_confidence:.3f} (Target: 0.7)")
    
    print("\n✅ Test complete")


if __name__ == "__main__":
    asyncio.run(test())