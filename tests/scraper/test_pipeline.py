"""
Test suite for Newsies Scraper Pipeline
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from newsies_scraper.pipelines.get_articles import get_articles_pipeline


class TestScraperPipeline:
    """Test cases for news scraping pipeline"""

    @pytest.fixture
    def mock_redis_task_status(self):
        """Mock Redis task status for testing"""
        with patch('newsies_scraper.pipelines.get_articles.redis_task_status') as mock:
            yield mock

    @pytest.fixture
    def mock_ap_news(self):
        """Mock AP News scraping functionality"""
        with patch('newsies_scraper.ap_news.latest_news.get_latest_news') as mock:
            yield mock

    @pytest.fixture
    def mock_article_loader(self):
        """Mock article loading functionality"""
        with patch('newsies_scraper.pipelines.get_articles.article_loader') as mock:
            yield mock

    @pytest.fixture
    def mock_article_ner(self):
        """Mock article NER functionality"""
        with patch('newsies_scraper.pipelines.get_articles.article_ner') as mock:
            yield mock

    @pytest.fixture
    def mock_article_formatter(self):
        """Mock article formatting functionality"""
        with patch('newsies_scraper.pipelines.get_articles.article_formatter') as mock:
            yield mock

    @pytest.fixture
    def mock_article_embeddings(self):
        """Mock article embeddings functionality"""
        with patch('newsies_scraper.pipelines.get_articles.article_embeddings') as mock:
            yield mock

    @pytest.fixture
    def mock_article_indexer(self):
        """Mock article indexing functionality"""
        with patch('newsies_scraper.pipelines.get_articles.article_indexer') as mock:
            yield mock

    def test_get_articles_pipeline_success(self, mock_redis_task_status, 
                                         mock_ap_news, mock_article_loader,
                                         mock_article_ner, mock_article_formatter,
                                         mock_article_embeddings, mock_article_indexer):
        """Test successful execution of get articles pipeline"""
        task_id = "test_scraper_task_123"
        
        # Mock successful execution of all steps
        mock_ap_news.return_value = True
        mock_article_loader.return_value = True
        mock_article_ner.return_value = True
        mock_article_formatter.return_value = True
        mock_article_embeddings.return_value = True
        mock_article_indexer.return_value = True
        
        # Execute pipeline
        get_articles_pipeline(task_id)
        
        # Verify all steps were called
        mock_ap_news.assert_called_once()
        mock_article_loader.assert_called_once()
        mock_article_ner.assert_called_once()
        mock_article_formatter.assert_called_once()
        mock_article_embeddings.assert_called_once()
        mock_article_indexer.assert_called_once()
        
        # Verify task status updates
        assert mock_redis_task_status.set_status.call_count >= 6  # Multiple status updates

    def test_get_articles_pipeline_failure_at_headlines(self, mock_redis_task_status, mock_ap_news):
        """Test pipeline failure at headlines retrieval step"""
        task_id = "test_scraper_task_fail"
        
        # Mock failure at first step
        mock_ap_news.side_effect = Exception("Failed to retrieve headlines")
        
        with pytest.raises(Exception) as exc_info:
            get_articles_pipeline(task_id)
        
        assert "Failed to retrieve headlines" in str(exc_info.value)
        
        # Verify error status was set
        mock_redis_task_status.set_status.assert_called()
        error_calls = [call for call in mock_redis_task_status.set_status.call_args_list 
                      if "error" in str(call)]
        assert len(error_calls) > 0

    def test_get_articles_pipeline_partial_failure(self, mock_redis_task_status,
                                                  mock_ap_news, mock_article_loader,
                                                  mock_article_ner):
        """Test pipeline partial failure at NER step"""
        task_id = "test_scraper_task_partial"
        
        # Mock success for first two steps, failure at NER
        mock_ap_news.return_value = True
        mock_article_loader.return_value = True
        mock_article_ner.side_effect = Exception("NER processing failed")
        
        with pytest.raises(Exception) as exc_info:
            get_articles_pipeline(task_id)
        
        assert "NER processing failed" in str(exc_info.value)
        
        # Verify first two steps completed
        mock_ap_news.assert_called_once()
        mock_article_loader.assert_called_once()
        mock_article_ner.assert_called_once()

    def test_pipeline_status_updates(self, mock_redis_task_status, mock_ap_news,
                                   mock_article_loader, mock_article_ner,
                                   mock_article_formatter, mock_article_embeddings,
                                   mock_article_indexer):
        """Test that pipeline updates task status at each step"""
        task_id = "test_status_updates"
        
        # Mock successful execution
        mock_ap_news.return_value = True
        mock_article_loader.return_value = True
        mock_article_ner.return_value = True
        mock_article_formatter.return_value = True
        mock_article_embeddings.return_value = True
        mock_article_indexer.return_value = True
        
        get_articles_pipeline(task_id)
        
        # Verify status updates for each step
        status_calls = mock_redis_task_status.set_status.call_args_list
        status_messages = [call[0][1] for call in status_calls]
        
        expected_statuses = [
            "started",
            "running - step: retrieving headlines",
            "running - step: retrieving and caching news articles",
            "running - step: detecting named entities in articles",
            "running - step: formatting article for LLM",
            "running - step: generating embeddings",
            "running - step: indexing articles",
            "complete"
        ]
        
        for expected_status in expected_statuses:
            assert any(expected_status in status for status in status_messages)

    @pytest.mark.integration
    def test_pipeline_with_real_data(self):
        """Integration test with real data structures"""
        # This would test with actual data structures but mocked external services
        task_id = "integration_test_task"
        
        with patch('newsies_scraper.pipelines.get_articles.redis_task_status') as mock_redis:
            with patch('newsies_scraper.ap_news.latest_news.get_latest_news') as mock_news:
                # Mock returning actual article data structure
                mock_news.return_value = {
                    "article_1": {"title": "Test Article", "url": "http://test.com"},
                    "article_2": {"title": "Test Article 2", "url": "http://test2.com"}
                }
                
                # Test would continue with real data flow...
                # For now, just verify the pipeline can be called
                try:
                    get_articles_pipeline(task_id)
                except Exception as e:
                    # Expected due to missing external services
                    assert "redis" in str(e).lower() or "connection" in str(e).lower()


class TestScraperUtilities:
    """Test utility functions in scraper package"""

    def test_validate_article_data(self):
        """Test article data validation"""
        from newsies_scraper.utils.validation import validate_article_data
        
        # Valid article data
        valid_article = {
            "title": "Test Article",
            "url": "http://example.com",
            "content": "Article content here",
            "published_date": "2023-01-01"
        }
        
        assert validate_article_data(valid_article) is True
        
        # Invalid article data
        invalid_article = {"title": ""}  # Missing required fields
        assert validate_article_data(invalid_article) is False

    def test_clean_article_content(self):
        """Test article content cleaning"""
        from newsies_scraper.utils.text_processing import clean_article_content
        
        dirty_content = "  Article content with\n\nextra whitespace  \n"
        clean_content = clean_article_content(dirty_content)
        
        assert clean_content == "Article content with extra whitespace"
        assert "\n\n" not in clean_content
        assert not clean_content.startswith(" ")
        assert not clean_content.endswith(" ")

    def test_extract_article_metadata(self):
        """Test article metadata extraction"""
        from newsies_scraper.utils.metadata import extract_article_metadata
        
        html_content = """
        <html>
            <meta property="og:title" content="Test Article Title">
            <meta property="og:description" content="Test description">
            <meta name="author" content="Test Author">
        </html>
        """
        
        metadata = extract_article_metadata(html_content)
        
        assert metadata["title"] == "Test Article Title"
        assert metadata["description"] == "Test description"
        assert metadata["author"] == "Test Author"


@pytest.mark.unit
class TestScraperErrorHandling:
    """Test error handling in scraper components"""

    def test_network_timeout_handling(self):
        """Test handling of network timeouts"""
        from newsies_scraper.ap_news.latest_news import get_latest_news
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = TimeoutError("Request timed out")
            
            result = get_latest_news()
            
            # Should handle timeout gracefully
            assert result is None or result == {}

    def test_invalid_html_handling(self):
        """Test handling of invalid HTML content"""
        from newsies_scraper.utils.html_parser import parse_article_content
        
        invalid_html = "<html><body><p>Unclosed paragraph"
        
        # Should not raise exception
        result = parse_article_content(invalid_html)
        assert isinstance(result, str)

    def test_missing_article_fields(self):
        """Test handling of articles with missing required fields"""
        from newsies_scraper.pipelines.get_articles import process_article
        
        incomplete_article = {"title": "Test"}  # Missing URL and content
        
        with pytest.raises(ValueError) as exc_info:
            process_article(incomplete_article)
        
        assert "missing required fields" in str(exc_info.value).lower()


@pytest.mark.slow
class TestScraperPerformance:
    """Performance tests for scraper components"""

    def test_large_article_processing(self):
        """Test processing of large articles"""
        from newsies_scraper.utils.text_processing import process_article_text
        
        # Create large article content (100KB)
        large_content = "This is a test article. " * 4000
        
        import time
        start_time = time.time()
        result = process_article_text(large_content)
        processing_time = time.time() - start_time
        
        # Should process within reasonable time (< 5 seconds)
        assert processing_time < 5.0
        assert len(result) > 0

    def test_concurrent_article_processing(self):
        """Test concurrent processing of multiple articles"""
        from newsies_scraper.utils.concurrent import process_articles_batch
        
        articles = [
            {"title": f"Article {i}", "content": f"Content {i}"} 
            for i in range(10)
        ]
        
        import time
        start_time = time.time()
        results = process_articles_batch(articles)
        processing_time = time.time() - start_time
        
        # Concurrent processing should be faster than sequential
        assert len(results) == 10
        assert processing_time < 10.0  # Should be much faster with concurrency
