from fourmp.cli import main


def test_main_prints_ready_message(capsys):
    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "4MP application is ready." in captured.out
