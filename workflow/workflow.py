# workflow.py
from langgraph.graph import StateGraph, START, END

from nodes.rss_node import rss_node
from nodes.summarizer_node import summarizer_node
from nodes.db_node import db_node
from nodes.embedding_node import embedding_node
from nodes.related_news_node import related_news_node
from nodes.update_db_node import update_db_node


def create_news_workflow():
    graph = StateGraph(state_schema=dict)

    # 1. Add node
    graph.add_node("rss", rss_node)
    graph.add_node("summarizer", summarizer_node)    
    graph.add_node("db", db_node)
    graph.add_node("embedding", embedding_node)
    graph.add_node("related_news", related_news_node)
    graph.add_node("update_db", update_db_node)

    # 2. Add edge
    graph.add_edge(START, "rss")
    graph.add_edge("rss", "summarizer")
    graph.add_edge("summarizer", "embedding")
    graph.add_edge("embedding", "db")
    graph.add_edge("db", 'related_news')
    graph.add_edge("related_news", "update_db")
    graph.add_edge("update_db", END)

    return graph.compile()