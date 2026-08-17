"""One-click GitHub publish helper.

Usage:
  python scripts/publish_github.py --token ghp_your_token_here
  python scripts/publish_github.py --token ghp_xxx --repo voice-rag-hh-goa-2026 --private

Get a token: https://github.com/settings/tokens/new
Required scopes: repo
"""
import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request


def create_github_repo(token: str, name: str, description: str, private: bool) -> dict:
    payload = json.dumps({
        "name": name,
        "description": description,
        "private": private,
        "auto_init": False,
        "has_issues": True,
        "has_wiki": False,
    }).encode()
    req = urllib.request.Request(
        "https://api.github.com/user/repos",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "voice-rag-publisher/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        data = json.loads(body)
        if e.code == 422 and "already exists" in body:
            print(f"⚠️  Repo already exists — skipping creation, will push to existing.")
            # Get existing repo
            req2 = urllib.request.Request(
                f"https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urllib.request.urlopen(req2) as r:
                user = json.loads(r.read())
            return {"clone_url": f"https://github.com/{user['login']}/{name}.git",
                    "html_url": f"https://github.com/{user['login']}/{name}"}
        raise RuntimeError(f"GitHub API error {e.code}: {body}") from e


def run(cmd: list, **kwargs):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return result


def main():
    parser = argparse.ArgumentParser(description="Publish Voice RAG to GitHub")
    parser.add_argument("--token", required=True, help="GitHub Personal Access Token (repo scope)")
    parser.add_argument("--repo", default="voice-rag-hh-goa-2026", help="Repository name")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    desc = (
        "Voice-enabled RAG over MSMARCO-XI for HH Goa 2026. "
        "ElevenLabs/Sarvam STT + ChromaDB + BM25 RRF + GPT-4o-mini. "
        "14 Indic languages. <200ms pipeline. 94 tests."
    )

    print(f"\n📦 Creating GitHub repo: {args.repo}...")
    repo = create_github_repo(args.token, args.repo, desc, args.private)
    clone_url = repo["clone_url"]
    html_url = repo["html_url"]
    print(f"✅ Repo URL: {html_url}")

    # Configure authenticated remote URL
    auth_url = clone_url.replace("https://", f"https://{args.token}@")

    # Set remote
    existing = run(["git", "remote", "-v"])
    if "origin" in existing.stdout:
        run(["git", "remote", "set-url", "origin", auth_url])
    else:
        run(["git", "remote", "add", "origin", auth_url])

    # Push
    print("\n🚀 Pushing to GitHub...")
    result = run(["git", "push", "-u", "origin", "master"])
    if result.returncode != 0:
        # Try main branch
        run(["git", "branch", "-M", "main"])
        result = run(["git", "push", "-u", "origin", "main"])

    if result.returncode == 0:
        print(f"\n🎉 Published successfully!")
        print(f"   Repository: {html_url}")
        print(f"   Clone:      git clone {clone_url}")
    else:
        print("\n❌ Push failed. Try manually:")
        print(f"   git remote add origin {clone_url}")
        print(f"   git push -u origin main")


if __name__ == "__main__":
    main()
