"""rdc shader edit-replay commands -- build, replace, restore shaders."""

from __future__ import annotations

import json
from pathlib import Path

import click

from rdc.commands._helpers import call, complete_eid, resolve_shader_target_eid
from rdc.formatters.json_fmt import write_json


@click.command("shader-encodings")
@click.option("--json", "use_json", is_flag=True, help="JSON output")
def shader_encodings_cmd(use_json: bool) -> None:
    """List available shader encodings for this capture."""
    result = call("shader_encodings", {})
    if use_json:
        write_json(result)
        return
    for enc in result["encodings"]:
        click.echo(enc["name"])


@click.command("shader-build")
@click.argument("source_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--stage",
    required=True,
    type=click.Choice(["vs", "hs", "ds", "gs", "ps", "cs"]),
    help="Shader stage",
)
@click.option("--entry", default="main", help="Entry point name")
@click.option("--encoding", default="GLSL", help="Encoding name (default: GLSL)")
@click.option("--json", "use_json", is_flag=True, help="JSON output")
@click.option("-q", "--quiet", is_flag=True, help="Print only shader_id")
def shader_build_cmd(
    source_file: Path, stage: str, entry: str, encoding: str, use_json: bool, quiet: bool
) -> None:
    """Build a shader from source file."""
    source = source_file.read_text("utf-8")
    result = call(
        "shader_build", {"stage": stage, "entry": entry, "encoding": encoding, "source": source}
    )
    if use_json:
        write_json(result)
        return
    if quiet:
        click.echo(result["shader_id"])
        return
    click.echo(f"shader_id\t{result['shader_id']}")
    if result.get("warnings"):
        click.echo(f"warnings\t{result['warnings']}", err=True)


@click.command("shader-replace")
@click.argument("eid", type=int, shell_complete=complete_eid)
@click.argument("stage", type=click.Choice(["vs", "hs", "ds", "gs", "ps", "cs"]))
@click.option(
    "--with", "shader_id", required=True, type=int, help="Built shader ID from shader-build"
)
@click.option("--json", "use_json", is_flag=True, help="JSON output")
def shader_replace_cmd(eid: int, stage: str, shader_id: int, use_json: bool) -> None:
    """Replace shader at EID/STAGE with a built shader."""
    result = call("shader_replace", {"eid": eid, "stage": stage, "shader_id": shader_id})
    if use_json:
        write_json(result)
        return
    click.echo(f"replaced\t{result['original_id']}")
    click.echo("warning: replacement affects all draws using this shader", err=True)


@click.command("shader-replace-target")
@click.argument("target", type=str)
@click.option(
    "--stage",
    required=True,
    type=click.Choice(["vs", "hs", "ds", "gs", "ps", "cs"]),
    help="Shader stage",
)
@click.option(
    "--with",
    "shader_id",
    required=True,
    type=int,
    help="Built shader ID from shader-build",
)
@click.option("--json", "use_json", is_flag=True, help="JSON output")
def shader_replace_target_cmd(target: str, stage: str, shader_id: int, use_json: bool) -> None:
    """Resolve TARGET from selected-event subtree first, then replace the shader."""
    eid, source, candidates = resolve_shader_target_eid(target, stage)
    if eid is None:
        if use_json:
            click.echo(
                json.dumps(
                    {
                        "error": {"message": f"unable to resolve target {target!r}"},
                        "source": source,
                        "candidates": candidates,
                    }
                ),
                err=True,
            )
            raise SystemExit(1)
        click.echo(f"error: unable to resolve target {target!r} ({source})", err=True)
        for cand in candidates[:5]:
            click.echo(
                f"candidate\t{cand.get('eid', 0)}\t{cand.get('action_name', '')}\t{cand.get('shader_name', '')}\t{cand.get('source', '')}",
                err=True,
            )
        raise SystemExit(1)

    result = call("shader_replace", {"eid": eid, "stage": stage, "shader_id": shader_id})
    if use_json:
        write_json({"resolved_eid": eid, "resolved_source": source, **result})
        return
    click.echo(f"resolved\t{eid}\t{source}")
    click.echo(f"replaced\t{result['original_id']}")
    click.echo("warning: replacement affects all draws using this shader", err=True)


@click.command("shader-restore")
@click.argument("eid", type=int, shell_complete=complete_eid)
@click.argument("stage", type=click.Choice(["vs", "hs", "ds", "gs", "ps", "cs"]))
@click.option("--json", "use_json", is_flag=True, help="JSON output")
def shader_restore_cmd(eid: int, stage: str, use_json: bool) -> None:
    """Restore original shader at EID/STAGE."""
    result = call("shader_restore", {"eid": eid, "stage": stage})
    if use_json:
        write_json(result)
        return
    click.echo(f"restored\t{stage}")


@click.command("shader-restore-all")
@click.option("--json", "use_json", is_flag=True, help="JSON output")
def shader_restore_all_cmd(use_json: bool) -> None:
    """Restore all replaced shaders and free built resources."""
    result = call("shader_restore_all", {})
    if use_json:
        write_json(result)
        return
    click.echo(f"restored\t{result['restored']}")
    click.echo(f"freed\t{result['freed']}")
