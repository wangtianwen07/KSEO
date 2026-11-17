"""GitHub PR/Issue 的有界只读检索。下载内容永远是数据，不执行命令。"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..util import KSEOError, digest, now, positive_int, redact, read_json, write_json

REPOSITORIES = {'ashhart/TensorFold', 'vllm-project/vllm'}
LINK = re.compile(r'https://github\.com/([^/]+/[^/]+)/(pull|issues)/(\d+)/?$')


def api(path: str):
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'KSEO-reference-research',
               'X-GitHub-Api-Version': '2022-11-28'}
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = 'Bearer ' + token
    try:
        with urlopen(Request('https://api.github.com' + path, headers=headers), timeout=30) as response:
            data = response.read(8_000_001)
        if len(data) > 8_000_000:
            raise KSEOError('GitHub 响应超过 8 MB；请缩小检索范围')
        return json.loads(data)
    except HTTPError as exc:
        raise KSEOError(f'GitHub 请求失败 HTTP {exc.code}；检查网络、配额或 GITHUB_TOKEN') from None
    except (URLError, ValueError, TimeoutError) as exc:
        raise KSEOError('GitHub 响应不可用；未生成完整快照') from exc


def cached_api(directory: Path, *, fetch=api, ttl_seconds=3600):
    """保留限流前的只读结果，避免重试耗尽同一批配额。缓存从不作为实测。"""
    def get(path):
        cache = directory / (digest(path) + '.json')
        if cache.is_file():
            entry = read_json(cache)
            if entry.get('path') == path and entry.get('sha256') == digest(entry.get('response')):
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(entry['retrieved_at'])).total_seconds()
                if 0 <= age < ttl_seconds:
                    return entry['response']
        response = fetch(path)
        write_json(cache, {'path': path, 'retrieved_at': now(), 'response': response, 'sha256': digest(response)})
        return response
    return get


def parse_link(url):
    match = LINK.fullmatch(url) if isinstance(url, str) else None
    if not match or match[1] not in REPOSITORIES:
        raise KSEOError('仅接收 TensorFold/vLLM 的 GitHub PR/Issue 链接')
    return match[1], match[2], int(match[3])


def collect(plan: dict, *, fetch=api) -> dict:
    if plan.get('schema_version') != 1 or plan.get('target') != 'gb10':
        raise KSEOError('参考检索计划要求 schema_version=1、target=gb10')
    limit = positive_int(plan.get('max_items', 12), 'max_items')
    if limit > 40 or len(plan.get('queries', [])) > 8:
        raise KSEOError('一次检索最多 40 条资料、8 个查询')
    links = list(plan.get('urls', []))
    for url in links:
        parse_link(url)
    if len(links) > limit:
        raise KSEOError('明确指定的链接超过 max_items')
    searches = []
    for query in plan.get('queries', []):
        repo = query.get('repo')
        terms = query.get('terms')
        if repo not in REPOSITORIES or not isinstance(terms, str) or not terms or len(terms) > 200:
            raise KSEOError('检索 query 需要允许的 repo 和短 terms')
        title_terms = query.get('title_terms', [])
        if not isinstance(title_terms, list) or any(not isinstance(t, str) or not t for t in title_terms):
            raise KSEOError('title_terms 必须是非空字符串列表')
        # 即使 terms 自带 GitHub qualifier，返回链接也必须属于明确的 repo。
        result = fetch('/search/issues?' + urlencode({'q': f'repo:{repo} {terms}',
                       'sort': 'updated', 'order': 'desc', 'per_page': min(limit, 30)}))
        if result.get('incomplete_results'):
            raise KSEOError('GitHub 检索结果不完整，请缩小查询')
        searches.append({'query': query, 'total_count': result['total_count'], 'response_sha256': digest(result)})
        for item in result['items']:
            link = item['html_url']
            if (parse_link(link)[0] == repo and link not in links and len(links) < limit
                    and all(term.lower() in item.get('title', '').lower() for term in title_terms)):
                links.append(link)
    records = []
    for url in dict.fromkeys(links):
        repo, kind, number = parse_link(url)
        endpoint = f'/repos/{repo}/' + ('pulls' if kind == 'pull' else 'issues') + f'/{number}'
        item = fetch(endpoint)
        record = {'id': f'{repo}#{number}', 'url': url, 'repo': repo, 'kind': kind,
                  'title': redact(item['title']), 'body': redact((item.get('body') or '')[:12000]),
                  'body_truncated': len(item.get('body') or '') > 12000,
                  'state': item['state'], 'updated_at': item['updated_at'],
                  'response_sha256': digest(item), 'evidence_status': 'REFERENCE_ONLY',
                  'hardware_match': 'UNVERIFIED', 'comments': []}
        if kind == 'pull':
            record.update({'merged': item.get('merged', False), 'head_sha': item['head']['sha'],
                           'base_sha': item['base']['sha'], 'merge_commit_sha': item.get('merge_commit_sha')})
            files = fetch(endpoint + '/files?per_page=100')
            record['files'] = [{'path': f['filename'], 'sha': f.get('sha'), 'status': f['status'],
                                'patch_excerpt': redact(f.get('patch', '')[:4000])} for f in files]
            record['files_truncated'] = item.get('changed_files', 0) > len(files)
        if item.get('comments', 0):
            comments = fetch(f'/repos/{repo}/issues/{number}/comments?per_page=20')
            record['comments'] = [{'url': c['html_url'], 'updated_at': c['updated_at'],
                                   'body': redact((c.get('body') or '')[:3000]), 'response_sha256': digest(c)}
                                  for c in comments]
            record['comments_truncated'] = item['comments'] > len(comments)
        record['sha256'] = digest(record)
        records.append(record)
    snapshot = {'schema_version': 1, 'target': 'gb10', 'created_at': now(), 'plan_sha256': digest(plan),
                'searches': searches, 'records': records, 'evidence_boundary':
                '仅参考快照；开放 PR、合并状态及外部性能声明均不是本机验证。摘录可能不完整。'}
    snapshot['sha256'] = digest(snapshot)
    return snapshot


def validate_snapshot(snapshot):
    if (not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1 or snapshot.get('target') != 'gb10'
            or snapshot.get('sha256') != digest({k: v for k, v in snapshot.items() if k != 'sha256'})):
        raise KSEOError('参考快照格式或 SHA256 不匹配')
    for row in snapshot.get('records', []):
        parse_link(row['url'])
        if row.get('sha256') != digest({k: v for k, v in row.items() if k != 'sha256'}):
            raise KSEOError('参考记录被修改')


def select(snapshot, workflow=None, limit=5):
    validate_snapshot(snapshot)
    hotspot = (workflow or {}).get('hotspot', {})
    terms = set(re.findall(r'[a-z0-9_]+', json.dumps(hotspot).lower()))
    def rank(row):
        tokens = set(re.findall(r'[a-z0-9_]+', (row['title'] + ' ' + row['body']).lower()))
        return (-len(terms & tokens), row['id'])
    selected = []
    for row in sorted(snapshot['records'], key=rank)[:limit]:
        # prompt 只放有界摘要；完整材料留在冻结快照中按需阅读。
        excerpt = {key: row[key] for key in ('id', 'url', 'repo', 'kind', 'state', 'updated_at', 'sha256',
                   'merged', 'head_sha', 'base_sha', 'merge_commit_sha', 'evidence_status', 'hardware_match') if key in row}
        excerpt.update({'title': row['title'][:200], 'body': row['body'][:1400],
                        'files': [{**f, 'patch_excerpt': f['patch_excerpt'][:300]} for f in row.get('files', [])[:2]],
                        'comments': [{**c, 'body': c['body'][:400]} for c in row.get('comments', [])[:1]],
                        'untrusted_reference': True, 'excerpt_only': True, 'application_requirements':
                        ['核对固定 base/head SHA 与本地源码', '检查是否已合入及依赖选项', 'GB10 重新编译、正确性与 E2E 复验']})
        selected.append(excerpt)
    return selected
