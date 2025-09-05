// popup.js

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyBQNd9c801KEupiKqZdV5AZJ8gHPoPkVaY';

  // Config handling (storage-backed)
  const apiHostInput = document.getElementById('apiHost');
  const apiPortInput = document.getElementById('apiPort');
  const saveBtn = document.getElementById('saveCfg');

  async function loadConfig() {
    return new Promise((resolve) => {
      chrome.storage.sync.get({ apiHost: 'http://localhost', apiPort: '5001' }, (cfg) => {
        resolve(cfg);
      });
    });
  }

  async function saveConfig(cfg) {
    return new Promise((resolve) => {
      chrome.storage.sync.set(cfg, () => resolve());
    });
  }

  function makeBaseURL(host, port) {
    if (!host) return 'http://localhost:5001';
    // If host already contains protocol and port, leave as is when port empty
    if (!port) return host;
    // Avoid double colons
    return host.endsWith(':') ? `${host}${port}` : `${host}:${port}`;
  }

  const cfg = await loadConfig();
  apiHostInput.value = cfg.apiHost;
  apiPortInput.value = cfg.apiPort;
  let API_BASE = makeBaseURL(cfg.apiHost, cfg.apiPort);

  saveBtn.addEventListener('click', async () => {
    const newCfg = { apiHost: apiHostInput.value.trim(), apiPort: apiPortInput.value.trim() };
    await saveConfig(newCfg);
    API_BASE = makeBaseURL(newCfg.apiHost, newCfg.apiPort);
    outputDiv.innerHTML = `<p style="color:#9fe29f;">Settings saved. Using API: ${API_BASE}</p>` + outputDiv.innerHTML;
  });

  // Ping API health first so users see connectivity issues early
  try {
    const h = await fetch(`${API_BASE}/health`, { method: 'GET' });
    if (!h.ok) {
      outputDiv.innerHTML = `<p style="color:#ff8888;">API health check failed: ${h.status}</p>` + outputDiv.innerHTML;
    }
  } catch (e) {
    outputDiv.innerHTML = `<p style="color:#ff8888;">API not reachable at ${API_BASE}. Check host/port in Settings.</p>` + outputDiv.innerHTML;
  }

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      outputDiv.innerHTML = `<div class="section-title">YouTube Video ID</div><p>${videoId}</p><p>Fetching comments...</p>`;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        outputDiv.innerHTML += "<p>No comments found for this video.</p>";
        return;
      }

      outputDiv.innerHTML += `<p>Fetched ${comments.length} comments. Performing sentiment analysis...</p>`;
      const predictions = await getSentimentPredictions(comments);

      if (predictions) {
        // Process the predictions to get sentiment counts and sentiment data
        const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
        const sentimentData = []; // For trend graph
        const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);
        predictions.forEach((item, index) => {
          sentimentCounts[item.sentiment]++;
          sentimentData.push({
            timestamp: item.timestamp,
            sentiment: parseInt(item.sentiment)
          });
        });

        // Compute metrics
        const totalComments = comments.length;
        const uniqueCommenters = new Set(comments.map(comment => comment.authorId)).size;
        const totalWords = comments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(word => word.length > 0).length, 0);
        const avgWordLength = (totalWords / totalComments).toFixed(2);
        const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);

        // Normalize the average sentiment score to a scale of 0 to 10
        const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

        // Add the Comment Analysis Summary section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Analysis Summary</div>
            <div class="metrics-container">
              <div class="metric">
                <div class="metric-title">Total Comments</div>
                <div class="metric-value">${totalComments}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Unique Commenters</div>
                <div class="metric-value">${uniqueCommenters}</div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Comment Length</div>
                <div class="metric-value">${avgWordLength} words</div>
              </div>
              <div class="metric">
                <div class="metric-title">Avg Sentiment Score</div>
                <div class="metric-value">${normalizedSentimentScore}/10</div>
              </div>
            </div>
          </div>
        `;

        // Add the Sentiment Analysis Results section with a placeholder for the chart
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Analysis Results</div>
            <p>See the pie chart below for sentiment distribution.</p>
            <div id="chart-container"></div>
          </div>`;

        // Fetch and display the pie chart inside the chart-container div
        await fetchAndDisplayChart(sentimentCounts);

        // Add the Sentiment Trend Graph section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Trend Over Time</div>
            <div id="trend-graph-container"></div>
          </div>`;

        // Fetch and display the sentiment trend graph
        await fetchAndDisplayTrendGraph(sentimentData);

        // Add the Word Cloud section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Wordcloud</div>
            <div id="wordcloud-container"></div>
          </div>`;

        // Fetch and display the word cloud inside the wordcloud-container div
        await fetchAndDisplayWordCloud(comments.map(comment => comment.text));

        // Add Quality Labels pie (MiniLM)
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Quality Labels (MiniLM)</div>
            <div id="quality-pie"></div>
          </div>`;
        await fetchAndDisplayQualityPie(comments);

        // Add Hybrid Relevance pie (Keywords + Embeddings)
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Hybrid Relevance</div>
            <div id="hybrid-pie"></div>
          </div>`;
        const vd = await fetchVideoDetails(videoId);
        await fetchAndDisplayHybridPie(comments, vd?.title || '', vd?.description || '');

        // Add GoEmotions pie
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">GoEmotions (Top Label)</div>
            <div id="goemo-pie"></div>
          </div>`;
        await fetchAndDisplayGoEmoPie(comments);

        // Add the top comments section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Top 25 Comments with Sentiments</div>
            <ul class="comment-list">
              ${predictions.slice(0, 25).map((item, index) => `
                <li class="comment-item">
                  <span>${index + 1}. ${item.comment}</span><br>
                  <span class="comment-sentiment">Sentiment: ${item.sentiment}</span>
                </li>`).join('')}
            </ul>
          </div>`;
      }
    } else {
      outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await response.json();
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp: timestamp, authorId: authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      outputDiv.innerHTML += "<p>Error fetching comments.</p>";
    }
    return comments;
  }

  async function fetchVideoDetails(videoId) {
    try {
      const resp = await fetch(`https://www.googleapis.com/youtube/v3/videos?part=snippet&id=${videoId}&key=${API_KEY}`);
      const data = await resp.json();
      const item = (data.items && data.items[0]) || null;
      if (!item) return null;
      return { title: item.snippet.title || '', description: item.snippet.description || '' };
    } catch (e) { return null; }
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_BASE}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      let text = await response.text();
      let result;
      try { result = JSON.parse(text); } catch (_) { result = null; }
      if (response.ok && result) {
        return result; // includes sentiment and timestamp
      }
      const msg = (result && result.error) ? result.error : `HTTP ${response.status}: ${text}`;
      throw new Error(msg);
    } catch (error) {
      console.error("Error fetching predictions:", error);
      outputDiv.innerHTML += `<p style="color:#ff8888;">Error fetching sentiment predictions: ${error.message}</p>`;
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_BASE}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch chart image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the chart-container div
      const chartContainer = document.getElementById('chart-container');
      chartContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching chart image:", error);
      outputDiv.innerHTML += "<p>Error fetching chart image.</p>";
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_BASE}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch word cloud image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the wordcloud-container div
      const wordcloudContainer = document.getElementById('wordcloud-container');
      wordcloudContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching word cloud image:", error);
      outputDiv.innerHTML += "<p>Error fetching word cloud image.</p>";
    }
  }

  async function fetchAndDisplayQualityPie(comments) {
    try {
      const response = await fetch(`${API_BASE}/quality_labels_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = url; img.style.width = '100%'; img.style.marginTop = '10px';
      document.getElementById('quality-pie').appendChild(img);
    } catch (e) {
      outputDiv.innerHTML += `<p style="color:#ff8888;">Quality pie error: ${e.message}</p>`;
    }
  }

  async function fetchAndDisplayHybridPie(comments, videoTitle, videoDescription) {
    try {
      const response = await fetch(`${API_BASE}/hybrid_relevance_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments, videoTitle, videoDescription })
      });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = url; img.style.width = '100%'; img.style.marginTop = '10px';
      document.getElementById('hybrid-pie').appendChild(img);
    } catch (e) {
      outputDiv.innerHTML += `<p style="color:#ff8888;">Hybrid pie error: ${e.message}</p>`;
    }
  }

  async function fetchAndDisplayGoEmoPie(comments) {
    try {
      const response = await fetch(`${API_BASE}/emotion_labels_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) throw new Error(await response.text());
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = url; img.style.width = '100%'; img.style.marginTop = '10px';
      document.getElementById('goemo-pie').appendChild(img);
    } catch (e) {
      outputDiv.innerHTML += `<p style=\"color:#ff8888;\">GoEmotions pie error: ${e.message}</p>`;
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const response = await fetch(`${API_BASE}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch trend graph image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the trend-graph-container div
      const trendGraphContainer = document.getElementById('trend-graph-container');
      trendGraphContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching trend graph image:", error);
      outputDiv.innerHTML += "<p>Error fetching trend graph image.</p>";
    }
  }
});
