document.addEventListener("DOMContentLoaded", function() {
    // Example JavaScript for "Check Eligibility" button
    const checkBtn = document.getElementById("checkEligibilityBtn");
    
    if (checkBtn) { // Check if the element exists in the DOM
      checkBtn.addEventListener("click", function() {
        // You could include logic here to evaluate user data
        // For demo purposes, we'll just pop up an alert
        alert("Your eligibility is being checked. Please wait...");
      });
    }
  });
  