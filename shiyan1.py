import os
import re
from typing import List, Dict, Set, Tuple

class Document:
    """文档类，表示一个文本文档"""
    def __init__(self, doc_id: str, content: str):
        self.doc_id = doc_id  # 文档ID（如文件名）
        self.content = content  # 文档内容
        self.terms = self._tokenize(content)  # 文档分词结果
    
    def _tokenize(self, text: str) -> List[str]:
        """改进的分词函数，保留连字符和撇号"""
        text = text.lower()
        # 保留字母、数字、连字符、撇号和空白字符
        text = re.sub(r"[^\w\s'-]", '', text)
        # 分割并去除空字符串
        return [term for term in text.split() if term]
    
    def contains_term(self, term: str) -> bool:
        """检查文档是否包含某个词项"""
        return term.lower() in self.terms
    
    def get_term_frequency(self, term: str) -> int:
        """获取词项在文档中的出现次数"""
        term = term.lower()
        return self.terms.count(term)

class DocumentCollection:
    """文档集合类，管理所有文档"""
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.total_docs = 0
        self.term_statistics: Dict[str, int] = {}  # 词项在整个集合中的出现次数
    
    def add_document(self, doc_id: str, content: str):
        """添加文档到集合"""
        doc = Document(doc_id, content)
        self.documents[doc_id] = doc
        self.total_docs += 1
        
        # 更新词项统计
        for term in set(doc.terms):  # 使用set避免重复计数
            self.term_statistics[term] = self.term_statistics.get(term, 0) + 1
    
    def load_from_directory(self, directory: str):
        """从目录加载所有txt文件，增强错误处理"""
        print(f"\nLoading documents from: {directory}")
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        loaded_files = 0
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                try:
                    # 尝试多种编码格式
                    for encoding in ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']:
                        try:
                            with open(filepath, 'r', encoding=encoding) as f:
                                content = f.read()
                            self.add_document(filename, content)
                            loaded_files += 1
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        print(f"Warning: Could not decode {filename} with any known encoding")
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        print(f"Successfully loaded {loaded_files} documents")
        print(f"Total unique terms: {len(self.term_statistics)}")
    
    def get_document_ids(self) -> List[str]:
        """获取所有文档ID"""
        return list(self.documents.keys())
    
    def show_term_stats(self, top_n=20):
        """显示词项统计"""
        if not self.term_statistics:
            print("No term statistics available")
            return
        
        sorted_terms = sorted(self.term_statistics.items(), 
                             key=lambda x: x[1], reverse=True)
        
        print("\nTop terms in collection:")
        for i, (term, count) in enumerate(sorted_terms[:top_n], 1):
            print(f"{i}. {term}: {count} occurrences")
    
    def show_sample_documents(self, n=3):
        """显示部分文档样本"""
        print("\nSample documents:")
        for i, (doc_id, doc) in enumerate(self.documents.items()):
            if i >= n:
                break
            print(f"\nDocument '{doc_id}':")
            print("First 50 terms:", doc.terms[:50])
            print("Content preview:", doc.content[:100].replace('\n', ' ') + "...")

class BooleanRetriever:
    """布尔检索基类"""
    def __init__(self, collection: DocumentCollection):
        self.collection = collection
    
    def search(self, query: str) -> List[str]:
        """执行查询，返回匹配的文档ID"""
        raise NotImplementedError("子类必须实现此方法")
    
    def _parse_query(self, query: str) -> List[Tuple[str, str]]:
        """解析布尔查询，返回(操作符, 词项)列表"""
        tokens = query.split()
        terms = []
        i = 0
        n = len(tokens)
        
        while i < n:
            token = tokens[i].upper()
            if token in ('AND', 'OR', 'NOT'):
                if i + 1 >= n:
                    raise ValueError(f"Missing term after operator: {token}")
                terms.append((token, tokens[i+1].lower()))
                i += 2
            else:
                # 默认为AND关系
                terms.append(('AND', token.lower()))
                i += 1
        return terms

class BruteForceRetriever(BooleanRetriever):
    """暴力检索实现"""
    def search(self, query: str) -> List[str]:
        """暴力搜索实现"""
        try:
            terms = self._parse_query(query)
        except ValueError as e:
            print(f"Query parsing error: {str(e)}")
            return []
        
        if not terms:
            return []
        
        results = []
        for doc_id, doc in self.collection.documents.items():
            if self._matches(doc, terms):
                results.append(doc_id)
        
        return sorted(results)
    
    def _matches(self, doc: Document, terms: List[Tuple[str, str]]) -> bool:
        """检查文档是否匹配查询条件"""
        if not terms:
            return False
            
        # 初始状态取决于第一个操作符
        first_op, first_term = terms[0]
        if first_op == 'NOT':
            result = not doc.contains_term(first_term)
        else:
            result = doc.contains_term(first_term)
        
        for op, term in terms[1:]:
            if op == 'AND':
                result = result and doc.contains_term(term)
            elif op == 'OR':
                result = result or doc.contains_term(term)
            elif op == 'NOT':
                result = result and not doc.contains_term(term)
        
        return result

class InvertedIndexRetriever(BooleanRetriever):
    """基于倒排索引的检索实现"""
    def __init__(self, collection: DocumentCollection):
        super().__init__(collection)
        self.inverted_index: Dict[str, Set[str]] = {}
        self._build_index()
    
    def _build_index(self):
        """构建倒排索引"""
        print("\nBuilding inverted index...")
        for doc_id, doc in self.collection.documents.items():
            for term in set(doc.terms):  # 使用set去除文档内重复词项
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(doc_id)
        print(f"Index built with {len(self.inverted_index)} unique terms")
    
    def search(self, query: str) -> List[str]:
        """使用倒排索引进行搜索"""
        try:
            terms = self._parse_query(query)
        except ValueError as e:
            print(f"Query parsing error: {str(e)}")
            return []
        
        if not terms:
            return []
        
        result = None
        
        for op, term in terms:
            docs = self.inverted_index.get(term, set())
            
            if op == 'AND':
                if result is None:
                    result = docs.copy()
                else:
                    result.intersection_update(docs)
            elif op == 'OR':
                if result is None:
                    result = docs.copy()
                else:
                    result.update(docs)
            elif op == 'NOT':
                all_docs = set(self.collection.documents.keys())
                if result is None:
                    result = all_docs - docs
                else:
                    result.difference_update(docs)
        
        return sorted(result) if result else []
    
    def check_term(self, term: str):
        """检查词项是否在索引中"""
        term = term.lower()
        if term in self.inverted_index:
            print(f"Term '{term}' found in {len(self.inverted_index[term])} documents")
            print("Sample documents containing this term:", 
                  list(self.inverted_index[term])[:3])
        else:
            print(f"Term '{term}' not found in index")

class SearchEngine:
    """搜索引擎主类"""
    def __init__(self, data_dir: str):
        self.collection = DocumentCollection()
        self.collection.load_from_directory(data_dir)
        self.brute_force_retriever = BruteForceRetriever(self.collection)
        self.inverted_index_retriever = InvertedIndexRetriever(self.collection)
    
    def search(self, query: str, method: str = 'inverted') -> List[str]:
        """执行搜索"""
        if method == 'brute':
            print(f"\nExecuting brute-force search for: {query}")
            return self.brute_force_retriever.search(query)
        else:
            print(f"\nExecuting inverted-index search for: {query}")
            return self.inverted_index_retriever.search(query)
    
    def analyze_query(self, query: str):
        """分析查询中的词项"""
        terms = self.inverted_index_retriever._parse_query(query)
        print("\nQuery analysis:")
        for op, term in terms:
            self.inverted_index_retriever.check_term(term)

if __name__ == "__main__":
    data_dir = "D:\\信息检索\\语料"  # 使用双反斜杠或原始字符串
    
    try:
        engine = SearchEngine(data_dir)
    except Exception as e:
        print(f"Failed to initialize search engine: {str(e)}")
        exit(1)
    
    # 显示统计信息和样本
    engine.collection.show_term_stats()
    engine.collection.show_sample_documents()
    
    # 测试查询
    test_queries = [
        "Brutus AND Caesar AND NOT Calpurnia"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"Testing query: {query}")
        
        # 分析查询词项
        engine.analyze_query(query)
        
        # 使用倒排索引检索
        print("\nInverted Index Results:")
        results = engine.search(query, method='inverted')
        print("\t" + "\n\t".join(results) if results else "\tNo results found")
        
        # 使用暴力检索
        print("\nBrute Force Results:")
        results = engine.search(query, method='brute')
        print("\t" + "\n\t".join(results) if results else "\tNo results found")