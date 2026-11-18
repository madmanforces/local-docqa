//! 데스크톱 창. 하는 일은 두 가지뿐이다.
//! 1) 창을 열 때 파이썬 엔진(`docqa serve`, 127.0.0.1:8765)을 뒤에서 띄운다.
//! 2) 창을 닫으면 그 엔진을 끈다.
//! 화면(ui/index.html)은 브라우저에서 쓰던 것과 같다. 답을 만드는 일은 전부 파이썬 엔진과 Ollama가 한다.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Manager, RunEvent};

const PORT: u16 = 8765;

struct Backend(Mutex<Option<Child>>);

/// 프로젝트 폴더(pyproject.toml이 있는 곳)를 찾는다.
/// DOCQA_ROOT 환경 변수 → 실행 파일 위치에서 위로 → 빌드한 위치 순서.
fn project_root() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DOCQA_ROOT") {
        return Some(PathBuf::from(p));
    }
    let is_root = |d: &PathBuf| d.join("pyproject.toml").exists() && d.join("docqa").is_dir();
    if let Some(mut dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf())) {
        loop {
            if is_root(&dir) {
                return Some(dir);
            }
            if !dir.pop() {
                break;
            }
        }
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    is_root(&dev).then_some(dev)
}

fn port_open() -> bool {
    TcpStream::connect_timeout(&([127, 0, 0, 1], PORT).into(), Duration::from_millis(300)).is_ok()
}

fn spawn_backend() -> Option<Child> {
    if port_open() {
        return None; // 이미 떠 있는 엔진(예: 직접 실행한 docqa serve)을 그대로 쓴다
    }
    let root = project_root()?;
    let venv_py = root.join(".venv").join("Scripts").join("python.exe");
    // --parent-pid: 앱이 어떻게 끝나든(강제 종료 포함) 엔진이 스스로 꺼지게 한다.
    // (.venv의 python.exe는 진짜 파이썬을 자식으로 띄우는 실행기라, 앱이 kill해도 자식이 남을 수 있다)
    let port = PORT.to_string();
    let me = std::process::id().to_string();
    let serve_args = ["serve", "--port", port.as_str(), "--parent-pid", me.as_str()];
    let mut cmd = if venv_py.exists() {
        let mut c = Command::new(venv_py);
        c.args(["-m", "docqa.cli"]).args(serve_args);
        c
    } else {
        let mut c = Command::new("uv");
        c.args(["run", "docqa"]).args(serve_args);
        c
    };
    cmd.current_dir(&root);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: 검은 콘솔 창을 띄우지 않는다
    }
    cmd.spawn().ok()
}

fn main() {
    let app = tauri::Builder::default()
        .setup(|app| {
            app.manage(Backend(Mutex::new(spawn_backend())));
            // 데모 GIF 녹화용: DOCQA_DEMO=1이면 화면이 질문을 자동으로 입력한다.
            if std::env::var("DOCQA_DEMO").is_ok() {
                if let Some(w) = app.get_webview_window("main") {
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_secs(4));
                        let _ = w.eval("window.startDemo && window.startDemo()");
                    });
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Tauri 앱을 만들지 못했습니다");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            if let Some(state) = handle.try_state::<Backend>() {
                if let Some(mut child) = state.0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        }
    });
}
