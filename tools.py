"""The tools the agents call. Each one logs its call for POST /chat's tool_calls.

The run's context object must have a `tool_calls` list.
"""

import asyncio
import json
from collections.abc import Callable

from agents import function_tool
from agents.tool_context import ToolContext

import data_tools
import facts
import wikipedia

MAX_QUERIES = 6  # per researcher


async def _logged(ctx: ToolContext, name: str, args: dict, call: Callable[[], str]) -> str:
    # In a thread, so parallel researchers do not block each other on a query or a web request.
    result = await asyncio.to_thread(call)
    ctx.context.tool_calls.append(
        {"name": name, "args": args, "result": result, "agent": ctx.agent.name})
    return result


@function_tool
async def run_sql(ctx: ToolContext, query: str) -> str:
    """Run one read-only SELECT on the MLB DuckDB database. Returns up to 200 rows as JSON, with
    any caveats about the data in `notes`, and a query_id such as Q3: cite a value from it as
    {Q3.r1.column}, where r1 is the first row. On failure, returns the error and a hint to fix it.

    Args:
        query: One DuckDB SELECT statement, e.g. 'SELECT home_team, avg(attendance) FROM games
            WHERE season = 2025 GROUP BY 1'. Call describe_schema first for table and column names.
    """
    run, agent = ctx.context, ctx.agent.name
    used = run.query_counts.get(agent, 0)
    run.query_counts[agent] = used + 1
    if used >= MAX_QUERIES:  # the prompt asks for at most 6; this holds it
        return await _logged(ctx, "run_sql", {"query": query}, lambda: json.dumps({
            "error": f"This researcher has used its {MAX_QUERIES} queries.",
            "hint": "Report what you have found, and say what is still missing."}))
    key = " ".join(query.lower().split())
    if key in run.query_cache:  # the same query again: the same answer, without running it
        cached = json.loads(run.query_cache[key])
        cached["note"] = "Already run: this is the same result as before. Do not run it again."
        return await _logged(ctx, "run_sql", {"query": query}, lambda: json.dumps(cached))
    # The query ID is added here, in the event loop, so parallel researchers never share one.
    result = facts.add_query_id(run, await asyncio.to_thread(data_tools.run_sql, query))
    run.tool_calls.append({"name": "run_sql", "args": {"query": query}, "result": result, "agent": agent})
    run.query_cache[key] = result
    return result


@function_tool
async def describe_schema(ctx: ToolContext, table: str = "") -> str:
    """List the database's tables and what the data covers, or the columns of one table.

    Args:
        table: A table name, e.g. 'games', to list its columns and their meaning. Leave empty to
            list all tables.
    """
    return await _logged(ctx, "describe_schema", {"table": table},
                         lambda: data_tools.describe_schema(table))


@function_tool
async def search_wikipedia(ctx: ToolContext, query: str, limit: int = 5) -> str:
    """Search English Wikipedia. Returns matching article titles, URLs and snippets as JSON.

    Args:
        query: What to look for. Team season articles work best, e.g. '2025 New York Mets season'.
        limit: How many results to return, from 1 to 10.
    """
    return await _logged(ctx, "search_wikipedia", {"query": query, "limit": limit},
                         lambda: wikipedia.search(query, limit))


@function_tool
async def read_wikipedia(ctx: ToolContext, title: str, offset: int = 0) -> str:
    """Read an English Wikipedia article as plain text, up to 12,000 characters per call. Cite the
    returned url.

    Args:
        title: The article's title or URL, taken from search_wikipedia results.
        offset: The character to start from. Use the offset given in `truncated` to read on from
            where the last call stopped. Leave at 0 for the start of the article.
    """
    return await _logged(ctx, "read_wikipedia", {"title": title, "offset": offset},
                         lambda: wikipedia.read(title, offset))
