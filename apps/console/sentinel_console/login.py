"""Who is at the console.

Two dialogs. `LoginDialog` asks a name and a password and returns the `User`
the store authenticates, or nothing after the attempts run out — a console
that let somebody through on a cancelled login would not be a lock.
`FirstAdminDialog` runs when the store holds no account at all: it offers to
create the first administrator and can be declined, because a deployment that
has not decided on accounts yet must still open — the status bar then says,
on every start, that nobody is named in the audit trail.

Neither dialog ever logs, echoes or keeps the password; it goes to
`Accounts.authenticate` and nowhere else.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from sentinel.accounts import AccountError, Accounts, Role, User

#: Wrong guesses before the dialog gives up. The store's own lockout counts
#: across dialogs and processes; this only bounds one sitting.
MAX_ATTEMPTS = 5


class LoginDialog(QDialog):
    def __init__(self, accounts: Accounts, parent: QWidget | None = None):
        super().__init__(parent)
        self._accounts = accounts
        self._user: User | None = None
        self._attempts = 0
        self.setWindowTitle("Sentinel Vision — sign in")
        self.setModal(True)

        layout = QVBoxLayout(self)
        caption = QLabel("Every change you make is written to the audit trail under your name.")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("account name")
        form.addRow("Name", self.name)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Password", self.password)
        layout.addLayout(form)
        self.message = QLabel("")
        self.message.setObjectName("Caption")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Sign in")
        buttons.accepted.connect(self._try)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.name.setFocus()

    def _try(self) -> None:
        try:
            self._user = self._accounts.authenticate(self.name.text(), self.password.text())
        except AccountError as error:
            self._attempts += 1
            self.password.clear()
            if self._attempts >= MAX_ATTEMPTS:
                self.reject()
                return
            self.message.setText(f"{error}. {MAX_ATTEMPTS - self._attempts} attempt(s) left.")
            return
        self.accept()

    @property
    def user(self) -> User | None:
        return self._user


class FirstAdminDialog(QDialog):
    """No account exists yet. Offer to create the first administrator."""

    def __init__(self, accounts: Accounts, parent: QWidget | None = None):
        super().__init__(parent)
        self._accounts = accounts
        self._user: User | None = None
        self.setWindowTitle("Sentinel Vision — first administrator")
        self.setModal(True)

        layout = QVBoxLayout(self)
        caption = QLabel(
            "No account exists yet. Create the first administrator so that every "
            "change to this site is recorded under a name. You can skip this: the "
            "console then opens with nothing gated and the audit trail names "
            "nobody, which the status bar will keep saying."
        )
        caption.setWordWrap(True)
        layout.addWidget(caption)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("one word")
        form.addRow("Name", self.name)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Password", self.password)
        self.confirm = QLineEdit()
        self.confirm.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow("Again", self.confirm)
        self.role = QComboBox()
        self.role.addItem("Administrator", Role.ADMIN.value)
        form.addRow("Role", self.role)
        layout.addLayout(form)
        self.message = QLabel("")
        self.message.setObjectName("Caption")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Skip for now")
        buttons.accepted.connect(self._create)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _create(self) -> None:
        if self.password.text() != self.confirm.text():
            self.message.setText("The two passwords differ.")
            self.confirm.clear()
            return
        if len(self.password.text()) < 8:
            self.message.setText("Use at least eight characters.")
            return
        try:
            self._user = self._accounts.add(
                self.name.text(), self.password.text(), Role(self.role.currentData()),
                actor="console:first-run",
            )
        except AccountError as error:
            self.message.setText(str(error))
            return
        self.accept()

    @property
    def user(self) -> User | None:
        return self._user
