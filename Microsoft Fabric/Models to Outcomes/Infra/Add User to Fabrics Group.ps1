# Assign the lab user account to the FabricLabs security group
Add-AzADGroupMember -TargetGroupDisplayName "SG Fabric Group" -MemberUserPrincipalName "Username"
Add-AzADGroupMember -TargetGroupDisplayName "LabFabricUsers" -MemberUserPrincipalName "Username"