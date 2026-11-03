//! Windows feedback directory protection. Keep `PrivateDirectory` alive while
//! SQLite is open; its handles deny rename/delete sharing on every ancestor.
use std::{
    ffi::c_void,
    io,
    mem::{size_of, zeroed},
    os::windows::ffi::OsStrExt,
    path::{Component, Path, PathBuf},
    ptr::{null, null_mut},
};
use windows_sys::Win32::{
    Foundation::{
        CloseHandle, ERROR_ALREADY_EXISTS, ERROR_FILE_NOT_FOUND, HANDLE, INVALID_HANDLE_VALUE,
        LocalFree,
    },
    Security::Authorization::{
        ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW,
        GetSecurityInfo, SE_FILE_OBJECT,
    },
    Security::{
        ACCESS_ALLOWED_ACE, ACL, CONTAINER_INHERIT_ACE, DACL_SECURITY_INFORMATION, EqualSid,
        GetAce, GetSecurityDescriptorControl, GetTokenInformation, INHERITED_ACE,
        OBJECT_INHERIT_ACE, OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID,
        SE_DACL_PROTECTED, SECURITY_ATTRIBUTES, TOKEN_QUERY, TOKEN_USER, TokenUser,
    },
    Storage::FileSystem::{
        BY_HANDLE_FILE_INFORMATION, CreateDirectoryW, CreateFileW, FILE_ALL_ACCESS,
        FILE_ATTRIBUTE_DIRECTORY, FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS,
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_READ_ATTRIBUTES, FILE_SHARE_READ, FILE_SHARE_WRITE,
        GetDriveTypeW, GetFileInformationByHandle, OPEN_EXISTING, READ_CONTROL,
    },
    System::Threading::{GetCurrentProcess, OpenProcessToken},
};

struct Handle(HANDLE);
impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}
// Handles are owned and valid until drop; Windows handles may be transferred across threads.
unsafe impl Send for Handle {}
unsafe impl Sync for Handle {}
struct Local(*mut c_void);
impl Drop for Local {
    fn drop(&mut self) {
        unsafe {
            LocalFree(self.0);
        }
    }
}

fn err(context: &str) -> String {
    format!("{context}: {}", io::Error::last_os_error())
}
fn wide(path: &Path) -> Result<Vec<u16>, String> {
    let mut v: Vec<u16> = path.as_os_str().encode_wide().collect();
    if v.contains(&0) {
        return Err("NUL in feedback path".into());
    }
    v.push(0);
    Ok(v)
}
fn open(path: &Path, read_control: bool) -> Result<Handle, io::Error> {
    let w = wide(path).map_err(io::Error::other)?;
    let h = unsafe {
        CreateFileW(
            w.as_ptr(),
            FILE_READ_ATTRIBUTES | if read_control { READ_CONTROL } else { 0 },
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            null(),
            OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
            null_mut(),
        )
    };
    if h == INVALID_HANDLE_VALUE {
        Err(io::Error::last_os_error())
    } else {
        Ok(Handle(h))
    }
}
fn info(handle: &Handle, directory: bool) -> Result<(), String> {
    let mut i: BY_HANDLE_FILE_INFORMATION = unsafe { zeroed() };
    if unsafe { GetFileInformationByHandle(handle.0, &mut i) } == 0 {
        return Err(err("cannot inspect feedback path"));
    }
    if i.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err("feedback path includes a reparse point".into());
    }
    if (i.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0) != directory {
        return Err("feedback path has an unexpected file type".into());
    }
    if !directory && i.nNumberOfLinks != 1 {
        return Err("feedback database file has multiple hard links".into());
    }
    Ok(())
}
fn reject_git_marker(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path.join(".git")) {
        Ok(_) => Err("feedback directory is inside a Git worktree".into()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("cannot inspect feedback Git boundary: {error}")),
    }
}
fn token_user() -> Result<Vec<usize>, String> {
    let mut h = null_mut();
    if unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut h) } == 0 {
        return Err(err("cannot open process token"));
    }
    let token = Handle(h);
    let mut len = 0;
    unsafe {
        GetTokenInformation(token.0, TokenUser, null_mut(), 0, &mut len);
    }
    if len < size_of::<TOKEN_USER>() as u32 {
        return Err(err("cannot size token user"));
    }
    let mut buf = vec![0usize; (len as usize).div_ceil(size_of::<usize>())];
    if unsafe { GetTokenInformation(token.0, TokenUser, buf.as_mut_ptr().cast(), len, &mut len) }
        == 0
    {
        return Err(err("cannot read token user"));
    }
    Ok(buf)
}
fn sid(buf: &[usize]) -> PSID {
    unsafe { (*(buf.as_ptr().cast::<TOKEN_USER>())).User.Sid }
}
fn create_private(path: &Path, user: PSID) -> Result<(), String> {
    let mut p = null_mut();
    if unsafe { ConvertSidToStringSidW(user, &mut p) } == 0 {
        return Err(err("cannot format user SID"));
    }
    let p_guard = Local(p.cast());
    let mut n = 0;
    unsafe {
        while *p.add(n) != 0 {
            n += 1;
        }
    }
    let identity = String::from_utf16(unsafe { std::slice::from_raw_parts(p, n) })
        .map_err(|_| "invalid user SID")?;
    let spec: Vec<u16> = format!("O:{identity}D:P(A;OICI;FA;;;{identity})\0")
        .encode_utf16()
        .collect();
    let mut sd: PSECURITY_DESCRIPTOR = null_mut();
    if unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(spec.as_ptr(), 1, &mut sd, null_mut())
    } == 0
    {
        return Err(err("cannot build feedback ACL"));
    }
    let _sd = Local(sd);
    let attrs = SECURITY_ATTRIBUTES {
        nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: sd,
        bInheritHandle: 0,
    };
    let w = wide(path)?;
    if unsafe { CreateDirectoryW(w.as_ptr(), &attrs) } == 0 {
        let e = io::Error::last_os_error();
        if e.raw_os_error() != Some(ERROR_ALREADY_EXISTS as i32) {
            return Err(format!("cannot create private feedback directory: {e}"));
        }
    }
    drop(p_guard);
    Ok(())
}
fn verify_acl(h: &Handle, user: PSID, directory: bool) -> Result<(), String> {
    let mut owner: PSID = null_mut();
    let mut acl: *mut ACL = null_mut();
    let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
    let status = unsafe {
        GetSecurityInfo(
            h.0,
            SE_FILE_OBJECT,
            OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
            &mut owner,
            null_mut(),
            &mut acl,
            null_mut(),
            &mut descriptor,
        )
    };
    if status != 0 {
        return Err(format!(
            "cannot read feedback ACL: {}",
            io::Error::from_raw_os_error(status as i32)
        ));
    }
    let _sd = Local(descriptor);
    if owner.is_null() || acl.is_null() || unsafe { EqualSid(owner, user) } == 0 {
        return Err("feedback owner or ACL is not current-user-only".into());
    }
    let mut control = 0;
    let mut rev = 0;
    if unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut rev) } == 0 {
        return Err(err("cannot inspect feedback ACL control"));
    }
    if directory && control & SE_DACL_PROTECTED == 0 {
        return Err("feedback directory inherits an untrusted ACL".into());
    }
    if unsafe { (*acl).AceCount } != 1 {
        return Err("feedback ACL must contain exactly one current-user entry".into());
    }
    let mut ace = null_mut();
    if unsafe { GetAce(acl, 0, &mut ace) } == 0 || ace.is_null() {
        return Err(err("cannot inspect feedback ACL entry"));
    }
    let a = unsafe { &*(ace.cast::<ACCESS_ALLOWED_ACE>()) };
    // ACCESS_ALLOWED_ACE_TYPE is zero. Reject deny, object, callback and unusual flags.
    let expected_flags = if directory {
        (OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE) as u8
    } else {
        INHERITED_ACE as u8
    };
    if a.Header.AceType != 0
        || a.Header.AceFlags != expected_flags
        || a.Mask != FILE_ALL_ACCESS
        || unsafe { EqualSid((&a.SidStart as *const u32).cast_mut().cast(), user) } == 0
    {
        return Err("feedback ACL is not the required current-user-only inheritable entry".into());
    }
    Ok(())
}

/// Acquires a stable, private directory. All parents must already exist; only the
/// final component is created. Existing objects are never re-permissioned.
/// Call `verify_files` before SQLite opens and again after WAL initialization.
/// Hold the returned guard for the entire lifetime of the SQLite connection.
pub(crate) struct PrivateDirectory {
    path: PathBuf,
    _handles: Vec<Handle>,
    user: Vec<usize>,
}
impl PrivateDirectory {
    pub(crate) fn open(path: &Path) -> Result<Self, String> {
        if !path.is_absolute() {
            return Err("feedback path must be absolute".into());
        }
        let mut components = path.components();
        match components.next() {
            Some(Component::Prefix(p))
                if matches!(
                    p.kind(),
                    std::path::Prefix::Disk(_) | std::path::Prefix::VerbatimDisk(_)
                ) => {}
            _ => return Err("feedback path must use a local drive".into()),
        }
        if !matches!(components.next(), Some(Component::RootDir)) {
            return Err("feedback path must be rooted".into());
        }
        for c in components {
            match c {
                Component::Normal(s) => {
                    let w: Vec<u16> = s.encode_wide().collect();
                    if w.is_empty()
                        || w.last() == Some(&(b'.' as u16))
                        || w.last() == Some(&(b' ' as u16))
                        || w.contains(&(b':' as u16))
                    {
                        return Err("ambiguous feedback path component".into());
                    }
                }
                _ => return Err("feedback path cannot contain relative components".into()),
            }
        }
        let parent = path.parent().ok_or("feedback path has no parent")?;
        if parent == path {
            return Err("feedback directory cannot be a drive root".into());
        }
        let user = token_user()?;
        let mut handles = Vec::new();
        let mut dirs: Vec<&Path> = parent.ancestors().collect();
        dirs.reverse();
        let root = dirs.first().ok_or("feedback path has no drive root")?;
        // DRIVE_REMOVABLE, DRIVE_FIXED, DRIVE_RAMDISK. Reject mapped network
        // drives, unknown drive types, and unavailable drives.
        if !matches!(unsafe { GetDriveTypeW(wide(root)?.as_ptr()) }, 2 | 3 | 6) {
            return Err("feedback path must use an available local drive".into());
        }
        for p in dirs {
            let h = open(p, false).map_err(|e| format!("cannot open feedback ancestor: {e}"))?;
            info(&h, true)?;
            reject_git_marker(p)?;
            handles.push(h);
        }
        // An existing final directory may itself be a worktree root. A missing
        // final component is acceptable, but an existing reparse point is not.
        match open(path, false) {
            Ok(h) => {
                info(&h, true)?;
                reject_git_marker(path)?;
            }
            Err(e) if e.raw_os_error() == Some(ERROR_FILE_NOT_FOUND as i32) => {}
            Err(e) => return Err(format!("cannot inspect feedback directory: {e}")),
        }
        create_private(path, sid(&user))?;
        let h = open(path, true).map_err(|e| format!("cannot open feedback directory: {e}"))?;
        info(&h, true)?;
        reject_git_marker(path)?;
        verify_acl(&h, sid(&user), true)?;
        handles.push(h);
        let guard = Self {
            path: path.to_owned(),
            _handles: handles,
            user,
        };
        guard.verify_files()?;
        Ok(guard)
    }
    pub(crate) fn verify_files(&self) -> Result<(), String> {
        for name in [
            "feedback.sqlite3",
            "feedback.sqlite3-wal",
            "feedback.sqlite3-shm",
        ] {
            match open(&self.path.join(name), true) {
                Ok(h) => {
                    info(&h, false)?;
                    verify_acl(&h, sid(&self.user), false)?;
                }
                Err(e) if e.raw_os_error() == Some(ERROR_FILE_NOT_FOUND as i32) => {}
                Err(e) => return Err(format!("cannot inspect feedback database file: {e}")),
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_git_directory_and_git_file_before_creation() {
        for directory_marker in [true, false] {
            let root = tempfile::tempdir().unwrap();
            let marker = root.path().join(".git");
            if directory_marker {
                std::fs::create_dir(&marker).unwrap();
            } else {
                std::fs::write(&marker, b"gitdir: elsewhere").unwrap();
            }
            let path = root.path().join("private");
            assert!(PrivateDirectory::open(&path).is_err());
            assert!(!path.exists());
        }
    }
    #[test]
    fn refuses_insecure_existing_database() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("private");
        let external = root.path().join("external");
        std::fs::write(&external, b"test").unwrap();
        let guard = PrivateDirectory::open(&path).unwrap();
        std::fs::rename(&external, path.join("feedback.sqlite3")).unwrap();
        assert!(guard.verify_files().is_err());
        drop(guard);
        assert!(PrivateDirectory::open(&path).is_err());
    }
    #[test]
    fn rejects_junction_ancestor() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("target");
        std::fs::create_dir(&target).unwrap();
        let junction = root.path().join("junction");
        let output = std::process::Command::new("cmd.exe")
            .args(["/C", "mklink", "/J"])
            .arg(&junction)
            .arg(&target)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "junction creation failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(PrivateDirectory::open(&junction.join("private")).is_err());
    }
    #[test]
    fn sqlite_wal_files_inherit_the_acl() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("private");
        let guard = PrivateDirectory::open(&path).unwrap();
        let db = rusqlite::Connection::open(path.join("feedback.sqlite3")).unwrap();
        db.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE sample (id INTEGER); INSERT INTO sample VALUES (1);").unwrap();
        assert!(path.join("feedback.sqlite3-wal").exists());
        assert!(path.join("feedback.sqlite3-shm").exists());
        guard.verify_files().unwrap();
    }
    #[test]
    fn creates_private_and_reopens() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("private");
        let guard = PrivateDirectory::open(&path).unwrap();
        std::fs::write(path.join("feedback.sqlite3"), b"test").unwrap();
        guard.verify_files().unwrap();
        drop(guard);
        PrivateDirectory::open(&path)
            .unwrap()
            .verify_files()
            .unwrap();
    }
    #[test]
    fn refuses_existing_inherited_acl_without_changing_it() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("insecure");
        std::fs::create_dir(&path).unwrap();
        assert!(PrivateDirectory::open(&path).is_err());
        assert!(path.exists());
    }
    #[test]
    fn refuses_hardlinked_database() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("private");
        let guard = PrivateDirectory::open(&path).unwrap();
        std::fs::write(path.join("feedback.sqlite3"), b"test").unwrap();
        std::fs::hard_link(path.join("feedback.sqlite3"), path.join("other")).unwrap();
        assert!(guard.verify_files().is_err());
    }
}
