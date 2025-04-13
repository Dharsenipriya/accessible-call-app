document.addEventListener("DOMContentLoaded", function() {
    // Initial posts data
    let posts = [
      {
        id: '1',
        title: 'Best practices for organic pest control',
        content: "I've been using neem oil with great success against aphids. Anyone else have organic solutions to share?",
        author: 'Sarah Johnson',
        likes: 12,
        comments: [
          {
            id: 'c1',
            content: 'I use a garlic and chili pepper spray that works well!',
            author: 'Tom Wilson',
            createdAt: new Date('2023-05-15')
          }
        ],
        createdAt: new Date('2023-05-10')
      },
      {
        id: '2',
        title: 'Crop rotation for small farms',
        content: 'Looking for advice on crop rotation patterns for my 5-acre vegetable farm. What has worked for others?',
        author: 'Miguel Rodriguez',
        likes: 8,
        comments: [],
        createdAt: new Date('2023-05-12')
      }
    ];
  
    let selectedPost = null;
  
    const postsSection = document.getElementById("posts-section");
    const postDetailSection = document.getElementById("post-detail-section");
  
    // Render posts list view
    function renderPosts() {
      postDetailSection.style.display = "none";
      postsSection.style.display = "block";
      let html = "";
  
      posts.forEach(post => {
        html += `
          <div class="card" data-id="${post.id}">
            <div class="card-header">
              <div class="card-title">${post.title}</div>
              <div class="card-description">${post.author} • ${post.createdAt.toLocaleDateString()}</div>
            </div>
            <div class="card-content">
              <p>${post.content.length > 100 ? post.content.substring(0, 100) + "..." : post.content}</p>
              <div>Likes: ${post.likes} | Comments: ${post.comments.length}</div>
            </div>
          </div>
        `;
      });
      postsSection.innerHTML = html;
  
      // Add click event to each card to view details
      document.querySelectorAll(".card").forEach(card => {
        card.addEventListener("click", function() {
          const postId = this.getAttribute("data-id");
          selectedPost = posts.find(p => p.id === postId);
          renderPostDetail();
        });
      });
    }
  
    // Render detailed post view
    function renderPostDetail() {
      postsSection.style.display = "none";
      postDetailSection.style.display = "block";
  
      let html = `
        <div class="detail-header">
          <button id="back-btn">← Back to all posts</button>
          <div class="detail-title">${selectedPost.title}</div>
          <div class="detail-meta">${selectedPost.author} • ${selectedPost.createdAt.toLocaleDateString()}</div>
        </div>
        <div class="detail-content">
          <p>${selectedPost.content}</p>
          <div>
            <button id="like-btn">Like (${selectedPost.likes})</button>
            <span>Comments: ${selectedPost.comments.length}</span>
          </div>
        </div>
        <div class="comments-section">
          <h3>Comments</h3>
      `;
      if (selectedPost.comments.length > 0) {
        selectedPost.comments.forEach(comment => {
          html += `
            <div class="comment-card">
              <div><strong>${comment.author}</strong> • ${comment.createdAt.toLocaleDateString()}</div>
              <div>${comment.content}</div>
            </div>
          `;
        });
      } else {
        html += `<p>No comments yet.</p>`;
      }
      html += `
        </div>
        <div class="add-comment">
          <textarea id="new-comment" placeholder="Add your comment..." rows="3"></textarea>
          <button id="post-comment-btn">Post Comment</button>
        </div>
      `;
      postDetailSection.innerHTML = html;
  
      // Back button event
      document.getElementById("back-btn").addEventListener("click", function() {
        renderPosts();
      });
  
      // Like button event
      document.getElementById("like-btn").addEventListener("click", function() {
        selectedPost.likes++;
        renderPostDetail();
      });
  
      // Add comment event
      document.getElementById("post-comment-btn").addEventListener("click", function() {
        const commentText = document.getElementById("new-comment").value.trim();
        if (commentText === "") return;
        const newComment = {
          id: Date.now().toString(),
          content: commentText,
          author: "Current User",
          createdAt: new Date()
        };
        selectedPost.comments.push(newComment);
        // Update the posts array
        posts = posts.map(post => post.id === selectedPost.id ? selectedPost : post);
        renderPostDetail();
      });
    }
  
    // Create new post event
    document.getElementById("create-post-btn").addEventListener("click", function() {
      const titleInput = document.getElementById("new-post-title");
      const contentInput = document.getElementById("new-post-content");
      const title = titleInput.value.trim();
      const content = contentInput.value.trim();
      if (title === "" || content === "") return;
      
      const newPost = {
        id: Date.now().toString(),
        title: title,
        content: content,
        author: "Current User",
        likes: 0,
        comments: [],
        createdAt: new Date()
      };
      
      posts.unshift(newPost);
      titleInput.value = "";
      contentInput.value = "";
      renderPosts();
    });
  
    // Initial render of posts list
    renderPosts();
  });
  