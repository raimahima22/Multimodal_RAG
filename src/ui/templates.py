# src/ui/templates.py
"""
All HTML snippet strings used by the Gradio UI.
Import these into main.py instead of defining them inline.
"""

APP_HEADER = """
<div id="app-header">
    <p id="app-wordmark">Benefits Assistant</p>
    <p id="app-tagline">Intelligent Q&amp;A over SBC &amp; SPD Documents</p>
</div>
"""

SIDEBAR = """
<div class="sidebar-label">Knowledge Base</div>

<div class="sidebar-doc">
    <div class="sidebar-doc-icon"><span>SBC</span></div>
    <div>
        <div class="sidebar-doc-name">Summary of Benefits</div>
        <div class="sidebar-doc-sub">Coverage &amp; Costs</div>
    </div>
</div>

<div class="sidebar-doc">
    <div class="sidebar-doc-icon"><span>SPD</span></div>
    <div>
        <div class="sidebar-doc-name">Plan Description</div>
        <div class="sidebar-doc-sub">Eligibility &amp; Rules</div>
    </div>
</div>

<div class="sidebar-divider"></div>

<div class="sidebar-stack-label">Powered By</div>
<div class="sidebar-stack-item">LangGraph Agent</div>
<div class="sidebar-stack-item">ColQwen2.5</div>
<div class="sidebar-stack-item">Qdrant Vector DB</div>

<div class="sidebar-footer">
    <span class="status-dot"></span>
    <span class="status-text">System ready</span>
</div>
"""

VOICE_HEADER = """
<div id="voice-header">
    <p id="voice-title">Voice-Enabled Assistant</p>
    <p id="voice-sub">
        Record your question — it will be transcribed, answered,
        and read back to you.
    </p>
</div>
"""