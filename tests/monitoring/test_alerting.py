#!/usr/bin/env python3
"""
Basic alerting test for CI workflow.
"""

import json
import sys


def test_alerting():
    """Test basic alerting mechanisms."""
    print("Testing alerting mechanisms...")
    
    # Simulate alert conditions and responses
    alerts = []
    
    # Test different alert scenarios
    test_conditions = [
        {"metric": "cpu_usage", "value": 85.0, "threshold": 80.0, "severity": "warning"},
        {"metric": "memory_usage", "value": 95.0, "threshold": 90.0, "severity": "critical"},
        {"metric": "disk_usage", "value": 70.0, "threshold": 80.0, "severity": "ok"},
        {"metric": "error_rate", "value": 0.05, "threshold": 0.01, "severity": "warning"}
    ]
    
    for condition in test_conditions:
        if condition["value"] > condition["threshold"]:
            alert = {
                "metric": condition["metric"],
                "value": condition["value"],
                "threshold": condition["threshold"],
                "severity": condition["severity"],
                "message": f"{condition['metric']} ({condition['value']}) exceeded threshold ({condition['threshold']})"
            }
            alerts.append(alert)
            print(f"🚨 ALERT: {alert['message']}")
        else:
            print(f"✓ {condition['metric']}: {condition['value']} (OK)")
    
    # Save alerts
    try:
        with open("alerts.json", "w") as f:
            json.dump(alerts, f, indent=2)
        
        print(f"✓ Alerting test completed. Generated {len(alerts)} alerts")
        return True
        
    except Exception as e:
        print(f"✗ Error in alerting test: {e}")
        return False


if __name__ == "__main__":
    if test_alerting():
        sys.exit(0)
    else:
        sys.exit(1)