import { pushToast } from './store';

/**
 * Put text on the clipboard. The browser clipboard call can be refused inside the app window (WebView2 asks no one and says no),
 * so a hidden text field and the older copy command are the second try. Resolves true when one of them worked.
 */
export async function copyText(value: string): Promise<boolean> {
    try {
        await navigator.clipboard.writeText(value);
        return true;
    }
    catch {
        // fall through to the older route
    }
    try {
        const field = document.createElement('textarea');
        field.value = value;
        field.setAttribute('readonly', '');
        field.style.cssText = 'position:fixed;left:-9999px;top:0;opacity:0';
        document.body.appendChild(field);
        field.select();
        const ok = document.execCommand('copy');
        field.remove();
        return ok;
    }
    catch {
        return false;
    }
}

/** Copy and say so: "Copied", or "Couldn't copy" with what to do instead. */
export async function copyAndTell(value: string, label: string): Promise<void> {
    if (await copyText(value))
        pushToast('success', 'Copied', label);
    else
        pushToast('warn', 'Couldn’t copy', 'Select the text and copy it manually.');
}
