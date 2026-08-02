"""Immutable research-data snapshots and deterministic replay gates.

The workflow is intentionally strict:

1. prepare market data once and freeze every causal input column;
2. hash market files, config, windows, and relevant source files;
3. run stop walk-forward only from the frozen snapshot;
4. replay the reference from the same snapshot and fail if any window/model
   differs beyond tolerance or changes its selected/executed variant.

Historical research only; no broker integration is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.backtest.portfolio_backtest_models import PortfolioBacktestConfig
from src.backtest.run_portfolio_backtest import prepare_portfolio_data
from src.backtest.run_portfolio_position_ablation import DEFAULT_TICKERS
from src.backtest.run_portfolio_stop_walk_forward import (
    MODEL_BASELINE, MODEL_LOW_DRAWDOWN, MODEL_MAX_RETURN, MODEL_RANK_WINNER,
    StopProfile, run_portfolio_stop_walk_forward, save_portfolio_stop_walk_forward,
)
from src.backtest.run_portfolio_walk_forward import build_walk_forward_windows
from src.backtest.run_portfolio_trade_timing_attribution import (
    DEFAULT_MODELS as DEFAULT_TIMING_MODELS, load_source as load_timing_source,
    run_trade_timing_attribution, save_trade_timing,
)

SCHEMA_VERSION = 1
DEFAULT_SNAPSHOT_ROOT = Path("data/backtests/portfolio/research_snapshots")
DEFAULT_WALK_FORWARD_OUTPUT = Path("data/backtests/portfolio/stop_walk_forward")
DEFAULT_GATE_OUTPUT = Path("data/backtests/portfolio/replay_gate")
CODE_PATHS = (
    "src/backtest/portfolio_backtest_models.py",
    "src/backtest/portfolio_backtest_engine.py",
    "src/backtest/run_backtest.py",
    "src/backtest/run_regime_ablation.py",
    "src/backtest/run_portfolio_backtest.py",
    "src/backtest/run_portfolio_position_ablation.py",
    "src/backtest/run_portfolio_walk_forward.py",
    "src/backtest/run_portfolio_stop_ablation.py",
    "src/backtest/run_portfolio_stop_walk_forward.py",
    "src/backtest/run_portfolio_trade_timing_attribution.py",
    "src/backtest/run_research_data_snapshot.py",
)


def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    if isinstance(value,float) and not isfinite(value):return None
    if isinstance(value,dict):return {str(key):_json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_json_safe(item) for item in value]
    return value


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output=frame.copy();output.index=pd.to_datetime(output.index)
    if output.index.tz is not None:output.index=output.index.tz_convert(None)
    return output.sort_index().loc[lambda value:~value.index.duplicated(keep="last")]


def _git_commit(project_root: Path) -> str | None:
    try:
        result=subprocess.run(["git","rev-parse","HEAD"],cwd=project_root,
            capture_output=True,text=True,check=True,timeout=10)
        value=result.stdout.strip();return value or None
    except (OSError,subprocess.SubprocessError):return None


def _code_hashes(project_root: Path) -> dict[str,str]:
    output={}
    for relative in CODE_PATHS:
        path=Path(project_root)/relative
        if path.exists():output[relative]=sha256_file(path)
    return output


def _write_market_frame(frame: pd.DataFrame, path: Path) -> dict[str,Any]:
    normalized=_normalize_frame(frame);output=normalized.copy()
    output.insert(0,"timestamp",output.index.strftime("%Y-%m-%dT%H:%M:%S"))
    output.to_csv(path,index=False,float_format="%.17g",lineterminator="\n")
    return {"path":str(path.name),"sha256":sha256_file(path),"rows":len(normalized),
        "start":normalized.index.min().isoformat(),"end":normalized.index.max().isoformat(),
        "columns":list(normalized.columns),"dtypes":{column:str(dtype) for column,dtype in normalized.dtypes.items()}}


def _restore_market_frame(path: Path, metadata: dict[str,Any]) -> pd.DataFrame:
    frame=pd.read_csv(path);timestamps=pd.to_datetime(frame.pop("timestamp"));frame.index=timestamps
    for column,dtype in metadata["dtypes"].items():
        if column not in frame:raise ValueError(f"Snapshot column missing: {column}")
        if dtype in ("bool","boolean"):
            frame[column]=frame[column].astype(str).str.lower().map({"true":True,"false":False})
            if frame[column].isna().any():raise ValueError(f"Invalid boolean values in {column}.")
        elif dtype.startswith(("float","int","uint")):
            frame[column]=pd.to_numeric(frame[column],errors="raise")
    return _normalize_frame(frame)


def create_snapshot_from_data(*, data_by_ticker: dict[str,pd.DataFrame],
        config: PortfolioBacktestConfig, output_root: Path=DEFAULT_SNAPSHOT_ROOT,
        project_root: Path=Path("."), period: str="provided", regime_period: str="provided",
        use_crypto_regime: bool=True, train_months: int=24, test_months: int=6,
        step_months: int=6) -> Path:
    """Freeze prepared data and return the final immutable snapshot directory."""
    if not data_by_ticker:raise ValueError("data_by_ticker cannot be empty.")
    config.validate();output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    project_root=Path(project_root).resolve()
    with tempfile.TemporaryDirectory(prefix="snapshot_build_",dir=output_root) as temporary:
        root=Path(temporary);market=root/"market";market.mkdir()
        market_files={}
        for ticker,frame in sorted(data_by_ticker.items()):
            safe=ticker.replace("/","_");path=market/f"{safe}.csv"
            market_files[ticker]=_write_market_frame(frame,path)
            market_files[ticker]["path"]=str(Path("market")/path.name)
        config_path=root/"config.json"
        config_path.write_text(json.dumps(_json_safe(config.to_dict()),sort_keys=True,indent=2)+"\n",encoding="utf-8")
        windows=build_walk_forward_windows(data_by_ticker,train_months=train_months,
            test_months=test_months,step_months=step_months)
        window_rows=[_json_safe(window.to_dict()) for window in windows]
        windows_path=root/"windows.csv";pd.DataFrame(window_rows).to_csv(windows_path,index=False,lineterminator="\n")
        core={"schema_version":SCHEMA_VERSION,"period":period,"regime_period":regime_period,
            "use_crypto_regime":bool(use_crypto_regime),"train_months":train_months,
            "test_months":test_months,"step_months":step_months,"tickers":sorted(data_by_ticker),
            "market_files":market_files,"config":{"path":"config.json","sha256":sha256_file(config_path)},
            "windows":{"path":"windows.csv","sha256":sha256_file(windows_path),"count":len(window_rows)},
            "code_files":_code_hashes(project_root),"git_commit":_git_commit(project_root)}
        fingerprint=sha256_json(core);created=datetime.now(UTC)
        snapshot_id=f"{created:%Y%m%d_%H%M%S}_{fingerprint[:12]}"
        manifest={**core,"snapshot_id":snapshot_id,"fingerprint":fingerprint,"created_at":created.isoformat()}
        (root/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        final=output_root/snapshot_id
        if final.exists():raise FileExistsError(final)
        shutil.move(str(root),str(final))
    return final


def create_snapshot(*, tickers: Iterable[str]=DEFAULT_TICKERS, period: str="10y",
        regime_period: str="max", use_crypto_regime: bool=True,
        config: PortfolioBacktestConfig|None=None, output_root: Path=DEFAULT_SNAPSHOT_ROOT,
        project_root: Path=Path("."), train_months: int=24, test_months: int=6,
        step_months: int=6) -> Path:
    active=config or PortfolioBacktestConfig()
    data=prepare_portfolio_data(tickers=list(tickers),period=period,regime_period=regime_period,
                                use_crypto_regime=use_crypto_regime)
    return create_snapshot_from_data(data_by_ticker=data,config=active,output_root=output_root,
        project_root=project_root,period=period,regime_period=regime_period,
        use_crypto_regime=use_crypto_regime,train_months=train_months,
        test_months=test_months,step_months=step_months)


def verify_snapshot(snapshot: Path, *, verify_code: bool=False,
                    project_root: Path=Path(".")) -> dict[str,Any]:
    snapshot=Path(snapshot);manifest_path=snapshot/"manifest.json"
    if not manifest_path.exists():raise FileNotFoundError(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"));rows=[]
    entries={"config":manifest["config"],"windows":manifest["windows"],
             **{f"market:{ticker}":metadata for ticker,metadata in manifest["market_files"].items()}}
    for label,metadata in entries.items():
        path=snapshot/metadata["path"];actual=sha256_file(path) if path.exists() else None
        rows.append({"artifact":label,"path":str(path),"expected_sha256":metadata["sha256"],
                     "actual_sha256":actual,"passed":actual==metadata["sha256"]})
    if verify_code:
        current=_code_hashes(Path(project_root).resolve())
        for relative,expected in manifest.get("code_files",{}).items():
            actual=current.get(relative);rows.append({"artifact":f"code:{relative}","path":relative,
                "expected_sha256":expected,"actual_sha256":actual,"passed":actual==expected})
    frame=pd.DataFrame(rows);return {"passed":bool(frame.passed.all()),"checks":frame,"manifest":manifest}


def load_snapshot(snapshot: Path, *, verify_code: bool=False,
                  project_root: Path=Path(".")) -> dict[str,Any]:
    verification=verify_snapshot(snapshot,verify_code=verify_code,project_root=project_root)
    if not verification["passed"]:
        failed=verification["checks"].loc[~verification["checks"].passed,"artifact"].tolist()
        raise ValueError("Snapshot verification failed: "+", ".join(failed))
    manifest=verification["manifest"];snapshot=Path(snapshot)
    config=PortfolioBacktestConfig(**json.loads((snapshot/manifest["config"]["path"]).read_text(encoding="utf-8")))
    data={ticker:_restore_market_frame(snapshot/metadata["path"],metadata)
          for ticker,metadata in manifest["market_files"].items()}
    return {"manifest":manifest,"config":config,"data_by_ticker":data,
            "windows":pd.read_csv(snapshot/manifest["windows"]["path"]),"verification":verification}


def _result_stamp(json_path: Path) -> str:
    prefix="portfolio_stop_walk_forward_";stem=Path(json_path).stem
    if not stem.startswith(prefix):raise ValueError(f"Unexpected result path: {json_path}")
    return stem[len(prefix):]


def _provenance_payload(snapshot:dict[str,Any],paths:dict[str,Path]) -> dict[str,Any]:
    return {"created_at":datetime.now(UTC).isoformat(),"schema_version":SCHEMA_VERSION,
        "snapshot_id":snapshot["manifest"]["snapshot_id"],
        "snapshot_fingerprint":snapshot["manifest"]["fingerprint"],
        "result_files":{name:{"path":str(path.resolve()),"sha256":sha256_file(path)} for name,path in paths.items()}}


def run_snapshot_walk_forward(*, snapshot_path:Path,
        output_directory:Path=DEFAULT_WALK_FORWARD_OUTPUT,
        stock_stops:Iterable[float]=(3.5,5.0,7.5),
        crypto_stops:Iterable[float]=(3.5,5.0,7.5)) -> dict[str,Any]:
    snapshot=load_snapshot(snapshot_path,verify_code=True)
    manifest=snapshot["manifest"]
    bundle=run_portfolio_stop_walk_forward(data_by_ticker=snapshot["data_by_ticker"],
        base_config=snapshot["config"],stock_stops=stock_stops,crypto_stops=crypto_stops,
        baseline_profile=StopProfile(5,5),rank_winner_profile=StopProfile(5,3.5),
        max_return_profile=StopProfile(3.5,3.5),low_drawdown_profile=StopProfile(7.5,3.5),
        train_months=int(manifest["train_months"]),test_months=int(manifest["test_months"]),
        step_months=int(manifest["step_months"]))
    paths=save_portfolio_stop_walk_forward(bundle,output_directory=Path(output_directory))
    stamp=_result_stamp(paths["json"]);provenance=_provenance_payload(snapshot,paths)
    provenance["source_stamp"]=stamp
    provenance["snapshot_manifest_path"]=str((Path(snapshot_path)/"manifest.json").resolve())
    provenance["snapshot_manifest_sha256"]=sha256_file(Path(snapshot_path)/"manifest.json")
    provenance_path=Path(output_directory)/f"portfolio_stop_walk_forward_provenance_{stamp}.json"
    provenance_path.write_text(json.dumps(provenance,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"bundle":bundle,"paths":{**paths,"provenance":provenance_path},"stamp":stamp,
            "snapshot":snapshot}


def compare_replay(reference:pd.DataFrame,replay:pd.DataFrame,
                   tolerance_percent:float=0.01) -> pd.DataFrame:
    if tolerance_percent<0:raise ValueError("tolerance_percent cannot be negative.")
    keys=["window_id","model"]
    required=set(keys+["variant_id","selected_variant_id","total_return_percent",
        "maximum_drawdown_percent","profit_factor","completed_trades"])
    for label,frame in (("reference",reference),("replay",replay)):
        missing=required.difference(frame.columns)
        if missing:raise ValueError(f"{label} missing: {sorted(missing)}")
        if frame.duplicated(keys).any():raise ValueError(f"{label} has duplicate window/model rows.")
    merged=reference[list(required)].merge(replay[list(required)],on=keys,how="outer",
        suffixes=("_reference","_replay"),indicator=True)
    rows=[]
    for row in merged.to_dict("records"):
        present=row["_merge"]=="both"
        def difference(name:str)->float|None:
            if not present:return None
            return abs(float(row[f"{name}_reference"])-float(row[f"{name}_replay"]))
        return_diff=difference("total_return_percent");drawdown_diff=difference("maximum_drawdown_percent")
        profit_diff=difference("profit_factor")
        variants=present and str(row["variant_id_reference"])==str(row["variant_id_replay"])
        selections=present and str(row["selected_variant_id_reference"])==str(row["selected_variant_id_replay"])
        trades=present and int(row["completed_trades_reference"])==int(row["completed_trades_replay"])
        numeric=present and return_diff<=tolerance_percent and drawdown_diff<=tolerance_percent and (
            profit_diff<=tolerance_percent or (not isfinite(float(row["profit_factor_reference"])) and
                                                not isfinite(float(row["profit_factor_replay"]))))
        rows.append({"window_id":row["window_id"],"model":row["model"],"present_in_both":present,
            "variant_match":variants,"selected_variant_match":selections,"trade_count_match":trades,
            "return_difference_percent":return_diff,"drawdown_difference_percent":drawdown_diff,
            "profit_factor_difference":profit_diff,"passed":bool(present and variants and selections and trades and numeric)})
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _profiles_from_reference(payload:dict[str,Any])->dict[str,StopProfile]:
    profiles=payload["profiles"]
    return {model:StopProfile(value["stock_stop_loss_percent"],value["crypto_stop_loss_percent"])
            for model,value in profiles.items()}


def replay_gate(*,snapshot_path:Path,reference_directory:Path,stamp:str,
                tolerance_percent:float=0.01) -> dict[str,Any]:
    snapshot=load_snapshot(snapshot_path,verify_code=True);reference_directory=Path(reference_directory)
    provenance_path=reference_directory/f"portfolio_stop_walk_forward_provenance_{stamp}.json"
    if not provenance_path.exists():raise FileNotFoundError(
        f"Unprovenanced reference is not eligible for replay gate: {provenance_path}")
    provenance=json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("snapshot_fingerprint")!=snapshot["manifest"]["fingerprint"]:
        raise ValueError("Reference and replay snapshot fingerprints differ.")
    payload=json.loads((reference_directory/f"portfolio_stop_walk_forward_{stamp}.json").read_text(encoding="utf-8"))
    profiles=_profiles_from_reference(payload)
    bundle=run_portfolio_stop_walk_forward(data_by_ticker=snapshot["data_by_ticker"],
        base_config=snapshot["config"],stock_stops=payload["stock_stops"],crypto_stops=payload["crypto_stops"],
        baseline_profile=profiles[MODEL_BASELINE],rank_winner_profile=profiles[MODEL_RANK_WINNER],
        max_return_profile=profiles[MODEL_MAX_RETURN],low_drawdown_profile=profiles[MODEL_LOW_DRAWDOWN],
        train_months=int(payload.get("train_months",snapshot["manifest"]["train_months"])),
        test_months=int(payload.get("test_months",snapshot["manifest"]["test_months"])),
        step_months=int(payload.get("step_months",snapshot["manifest"]["step_months"])))
    reference=pd.read_csv(reference_directory/f"portfolio_stop_walk_forward_test_runs_{stamp}.csv")
    checks=compare_replay(reference,bundle["test_runs"],tolerance_percent)
    return {"passed":bool(checks.passed.all()),"checks":checks,"bundle":bundle,
            "snapshot_id":snapshot["manifest"]["snapshot_id"],"snapshot_fingerprint":snapshot["manifest"]["fingerprint"],
            "reference_stamp":stamp,"tolerance_percent":tolerance_percent}


def save_gate(result:dict[str,Any],output_directory:Path=DEFAULT_GATE_OUTPUT)->dict[str,Path]:
    output_directory.mkdir(parents=True,exist_ok=True);stamp=datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    csv_path=output_directory/f"portfolio_replay_gate_checks_{stamp}.csv"
    json_path=output_directory/f"portfolio_replay_gate_{stamp}.json"
    result["checks"].to_csv(csv_path,index=False)
    payload={key:value for key,value in result.items() if key not in ("checks","bundle")}
    payload["failed_checks"]=result["checks"].loc[~result["checks"].passed].to_dict("records")
    json_path.write_text(json.dumps(_json_safe(payload),indent=2)+"\n",encoding="utf-8")
    return {"json":json_path,"checks":csv_path}


def run_snapshot_timing(*, snapshot_path:Path, reference_directory:Path, stamp:str,
        models:Iterable[str]=DEFAULT_TIMING_MODELS, horizons:Iterable[int]=(5,10,20),
        output_directory:Path=Path("data/backtests/portfolio/trade_timing_attribution")) -> dict[str,Any]:
    snapshot=load_snapshot(snapshot_path,verify_code=True);reference_directory=Path(reference_directory)
    provenance_path=reference_directory/f"portfolio_stop_walk_forward_provenance_{stamp}.json"
    if not provenance_path.exists():raise FileNotFoundError(
        f"Timing attribution requires a provenanced reference: {provenance_path}")
    provenance=json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("snapshot_fingerprint")!=snapshot["manifest"]["fingerprint"]:
        raise ValueError("Reference and timing snapshot fingerprints differ.")
    source=load_timing_source(reference_directory,stamp)
    result=run_trade_timing_attribution(source=source,data_by_ticker=snapshot["data_by_ticker"],
        models=tuple(models),horizons=tuple(horizons),bull_threshold=10,bear_threshold=-10)
    paths=save_trade_timing(result,Path(output_directory))
    timing_stamp=Path(paths["json"]).stem.removeprefix("portfolio_trade_timing_")
    timing_provenance={"created_at":datetime.now(UTC).isoformat(),"timing_stamp":timing_stamp,
        "source_stamp":stamp,"snapshot_id":snapshot["manifest"]["snapshot_id"],
        "snapshot_fingerprint":snapshot["manifest"]["fingerprint"],
        "result_files":{name:{"path":str(path.resolve()),"sha256":sha256_file(path)} for name,path in paths.items()}}
    timing_provenance_path=Path(output_directory)/f"portfolio_trade_timing_provenance_{timing_stamp}.json"
    timing_provenance_path.write_text(json.dumps(timing_provenance,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"result":result,"paths":{**paths,"provenance":timing_provenance_path},
            "timing_stamp":timing_stamp}


def _base_config(args:argparse.Namespace)->PortfolioBacktestConfig:
    return PortfolioBacktestConfig(initial_cash=args.initial_cash,risk_per_trade_percent=args.risk,
        maximum_position_percent=args.max_position,maximum_total_open_risk_percent=args.max_total_risk,
        maximum_crypto_allocation_percent=args.max_crypto,maximum_open_positions=args.max_positions,
        stock_stop_loss_percent=5,crypto_stop_loss_percent=5,stock_trailing_close_percent=7.5,
        crypto_trailing_close_percent=7.5,commission_rate=args.commission_rate,
        minimum_fee=args.minimum_fee,slippage_bps=args.slippage_bps)


def main():
    parser=argparse.ArgumentParser(description="Immutable research data and replay gates")
    sub=parser.add_subparsers(dest="command",required=True)
    create=sub.add_parser("create");create.add_argument("tickers",nargs="*");create.add_argument("--period",default="10y")
    create.add_argument("--regime-period",default="max");create.add_argument("--output-root",type=Path,default=DEFAULT_SNAPSHOT_ROOT)
    create.add_argument("--train-months",type=int,default=24);create.add_argument("--test-months",type=int,default=6);create.add_argument("--step-months",type=int,default=6)
    create.add_argument("--initial-cash",type=float,default=10000);create.add_argument("--risk",type=float,default=1)
    create.add_argument("--max-position",type=float,default=25);create.add_argument("--max-total-risk",type=float,default=4)
    create.add_argument("--max-crypto",type=float,default=25);create.add_argument("--max-positions",type=int,default=4)
    create.add_argument("--commission-rate",type=float,default=.0005);create.add_argument("--minimum-fee",type=float,default=1)
    create.add_argument("--slippage-bps",type=float,default=5);create.add_argument("--no-crypto-regime",action="store_true")
    verify=sub.add_parser("verify");verify.add_argument("--snapshot",type=Path,required=True);verify.add_argument("--verify-code",action="store_true")
    walk=sub.add_parser("walk-forward");walk.add_argument("--snapshot",type=Path,required=True);walk.add_argument("--output-directory",type=Path,default=DEFAULT_WALK_FORWARD_OUTPUT)
    gate=sub.add_parser("gate");gate.add_argument("--snapshot",type=Path,required=True);gate.add_argument("--reference-directory",type=Path,default=DEFAULT_WALK_FORWARD_OUTPUT)
    gate.add_argument("--stamp",required=True);gate.add_argument("--tolerance",type=float,default=.01);gate.add_argument("--output-directory",type=Path,default=DEFAULT_GATE_OUTPUT)
    timing=sub.add_parser("timing");timing.add_argument("--snapshot",type=Path,required=True)
    timing.add_argument("--reference-directory",type=Path,default=DEFAULT_WALK_FORWARD_OUTPUT)
    timing.add_argument("--stamp",required=True);timing.add_argument("--models",nargs="+",default=list(DEFAULT_TIMING_MODELS))
    timing.add_argument("--horizons",nargs="+",type=int,default=[5,10,20])
    timing.add_argument("--output-directory",type=Path,default=Path("data/backtests/portfolio/trade_timing_attribution"))
    args=parser.parse_args()
    if args.command=="create":
        path=create_snapshot(tickers=args.tickers or DEFAULT_TICKERS,period=args.period,regime_period=args.regime_period,
            use_crypto_regime=not args.no_crypto_regime,config=_base_config(args),output_root=args.output_root,
            train_months=args.train_months,test_months=args.test_months,step_months=args.step_months)
        print("SNAPSHOT",path.resolve())
    elif args.command=="verify":
        result=verify_snapshot(args.snapshot,verify_code=args.verify_code);print(result["checks"].to_string(index=False));print("PASS" if result["passed"] else "FAIL")
        if not result["passed"]:raise SystemExit(2)
    elif args.command=="walk-forward":
        result=run_snapshot_walk_forward(snapshot_path=args.snapshot,output_directory=args.output_directory)
        for name,path in result["paths"].items():print(name,path.resolve())
    elif args.command=="gate":
        result=replay_gate(snapshot_path=args.snapshot,reference_directory=args.reference_directory,
            stamp=args.stamp,tolerance_percent=args.tolerance)
        for name,path in save_gate(result,args.output_directory).items():print(name,path.resolve())
        print("PASS" if result["passed"] else "FAIL")
        if not result["passed"]:raise SystemExit(2)
    else:
        result=run_snapshot_timing(snapshot_path=args.snapshot,reference_directory=args.reference_directory,
            stamp=args.stamp,models=args.models,horizons=args.horizons,output_directory=args.output_directory)
        for name,path in result["paths"].items():print(name,path.resolve())


if __name__=="__main__":main()
